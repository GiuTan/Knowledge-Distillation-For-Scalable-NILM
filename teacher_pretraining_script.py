from network.CRNN_custom import *
from network.CRNN import *
from utils_func import *
import random as python_random
import tensorflow as tf
import network.metrics_losses
import os
from sklearn.metrics import  classification_report, multilabel_confusion_matrix
from metrics import *
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import argparse
from data.ukdale_loading import *


random.seed(123)
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)
tf.experimental.numpy.random.seed(1234)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

os.environ["CUDA_VISIBLE_DEVICES"]="6"


# Flag Inizialization
classes = 6
model_save = False
test = True
train = True
check_data =True
weak_lab = True
strong = False
strong_weak = True
var_weak = False
dataset = 'REFIT'

if dataset =='REFIT':
    file_agg_path = '/raid/users/eprincipi/KD_agg_REFIT_pretrain/'
    file_labels_path = '/raid/users/eprincipi/KD_labels_REFIT_pretrain/'
else: #UKDALE
    file_agg_path =  '/raid/users/eprincipi/KD_agg_UKDALE/'
    file_labels_path = '/raid/users/eprincipi/KD_labels_UKDALE/'

X_val = np.load(file_agg_path + 'new_X_val.npy')
Y_val = np.load(file_labels_path + 'new_Y_val.npy')
Y_val_weak = np.load(file_labels_path + 'new_Y_val_weak.npy')

X_train = np.load(file_agg_path + 'new_X_train.npy')
Y_train = np.load(file_labels_path + 'new_Y_train.npy')
Y_train_weak = np.load(file_labels_path + 'new_Y_train_weak.npy')

weak_X_train = np.load('/raid/users/eprincipi/KD_agg_REFIT_pretrain_weak/new_X_train_weak.npy')
weak_Y_train = np.negative(np.ones((len(weak_X_train),2550,6))) #np.load('/raid/users/eprincipi/KD_labels_UKDALE_weak/new_Y_train_WEAK.npy')
weak_Y_train_weak = np.load('/raid/users/eprincipi/KD_labels_REFIT_pretrain_weak/new_Y_train_weak_WEAK.npy')

X_train = np.concatenate([X_train,weak_X_train], axis=0)
Y_train = np.concatenate([Y_train,weak_Y_train], axis = 0)
Y_train_weak = np.concatenate([Y_train_weak,weak_Y_train_weak], axis = 0)
print(X_train.shape)
print(Y_train.shape)
print(Y_train_weak.shape)
Y_val = np.where(Y_val>0.5, 1, 0 )
Y_train = np.where(Y_train>0.5, 1, 0 )

assert(len(Y_val)==len(Y_val_weak))
assert(len(Y_train)==len(Y_train_weak))

# REFIT TEST DATA #
X_test_r = np.load('/raid/users/eprincipi/KD_agg_REFIT/new2_X_test.npy')
Y_test_r = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_test.npy')
Y_test_weak_r = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_test_weak.npy')
val_l = round(len(X_test_r) / 100 * 10 / 2)
X_test_r = X_test_r[val_l:-val_l]
Y_test_r = Y_test_r[val_l:-val_l]
Y_test_weak_r = Y_test_weak_r[val_l:-val_l]
Y_test_r = np.where(Y_test_r >0.5, 1, 0 )


x_train = X_train
y_strong_train = Y_train
y_weak_train = Y_train_weak

print('Data shape:')
print(x_train.shape)
print(X_val.shape)

# Aggregate Standardization #

train_mean = np.mean(x_train)
train_std = np.std(x_train)
print("Mean train")
print(train_mean)
print("Std train")
print(train_std)


x_train = standardize_data(x_train,train_mean, train_std)
X_val = standardize_data(X_val, train_mean, train_std)
X_test_r  =standardize_data(X_test_r, train_mean, train_std)


type_ = 'pretrained_'+ dataset

tuner = True
if tuner:
        batch_size = 64
        window_size = 2550
        drop_out  = 0.1
        kernel = 5
        num_layers = 3
        gru_units = 64
        lr = 0.002


        weight= 1e-2
        gamma = K.variable(1.0)
        weight_dyn = WeightAdjuster(weights=gamma)

        CRNN = CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop_out, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=None,strong_weak_flag=True, temperature=1.0)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_F1_score', mode='max',
                                                          patience=15, restore_best_weights=True)

        log_dir_ = '/logs/pretraining_'  + datetime.now().strftime("%Y%m%d-%H%M%S") + type_ + str(weight)
        tensorboard = TensorBoard(log_dir=log_dir_)
        file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
        file_writer.set_as_default()
        MODEL = CRNN_custom(CRNN)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        MODEL.compile(student_optimizer=optimizer, teacher_optimizer=optimizer,
                      loss={"strong_loss": network.metrics_losses.binary_crossentropy, "weak_loss": network.metrics_losses.binary_crossentropy_weak},
                      loss_weights=[weight], Temperature=1, F1_score=StatefullF1())

        MODEL.teacher.load_weights('/raid/users/eprincipi/Knowledge_Distillation/pretrained/CRNN_model_PRETRAINED_REFIT_SW_correct_bin0.01.h5')

        test = True
        if test:

                # prediction strong-weak
                out, output_strong, output_weak = MODEL.predict(x=X_val)

                outp, output_strong_train, output_weak_train = MODEL.predict(x=X_test_r)

                # prediction strong
                # output_strong = TCN.predict(x=X_val)
                # output_strong_test  = TCN.predict(x=X_test)
                # output_strong_train = TCN.predict(x=X_train_new)

                print(Y_val.shape)
                print(output_strong.shape)

                # reshaping
                shape = output_strong.shape[0] * output_strong.shape[1]

                shape_train = output_strong_train.shape[0] * output_strong_train.shape[1]

                Y_test_r = Y_test_r.reshape(shape_train,classes)
                # reshape both
                Y_val = Y_val.reshape(shape, classes)


                output_strong = output_strong.reshape(shape, classes)
                output_strong_train = output_strong_train.reshape(shape_train, classes)


                # calcolo i threshold
                thres_strong = thres_analysis(Y_val, output_strong,classes)



                # Y_train_weak = Y_train_weak.reshape(Y_train_weak.shape[0]*Y_train_weak.shape[1],classes)
                # output_weak = output_weak.reshape(output_weak.shape[0] * output_weak.shape[1], classes)
                # print(output_weak)
                # thres_weak = [0.501, 0.501, 0.501, 0.501, 0.501, 0.501]

                print(Y_val.shape)
                print(output_strong.shape)
                assert (Y_val.shape == output_strong.shape)
                binarization = True
                if binarization:


                    plt.plot(output_strong[:24000, 0])
                    plt.plot(Y_val[:24000, 0])
                    plt.legend(['output', 'y'])
                    plt.show()
                    plt.plot(output_strong[:24000, 1])
                    plt.plot(Y_val[:24000, 1])
                    plt.legend(['output', 'y'])
                    plt.show()

                    plt.plot(output_strong[:24000, 2])
                    plt.plot(Y_val[:24000, 2])
                    plt.legend(['output', 'y'])
                    plt.show()

                    plt.plot(output_strong[:24000, 3])
                    plt.plot(Y_val[:24000, 3])
                    plt.legend(['output', 'y'])
                    plt.show()

                    plt.plot(output_strong[:24000, 4])
                    plt.plot(Y_val[:24000, 4])
                    plt.legend(['output', 'y'])
                    plt.show()

                    ### STRONG

                    print(thres_strong)


                    # output_strong_test = app_binarization_strong(output_strong_test, thres_strong, classes)
                    output_strong = app_binarization_strong(output_strong, thres_strong, classes)
                    output_strong_train = app_binarization_strong(output_strong_train, thres_strong, classes)

                    ####  WEAK

                    #output_weak = app_binarization_weak(output_weak, thres_weak, classes)
                    # output_weak_test = app_binarization_weak(output_weak_test, thres_weak, classes)
                    #output_weak_train = app_binarization_weak(output_weak_train, thres_weak, classes)
                    #Y_train_weak = app_binarization_weak(Y_train_weak, thres_weak, classes)


                    #Y_val_weak = Y_val_weak.reshape(Y_val_weak.shape[0] * Y_val_weak.shape[1], classes)

                    #Y_train_weak = Y_train_weak.reshape(Y_train_weak.shape[0] * Y_train_weak.shape[1], 5)

                    # plt.plot(output_strong[:24000, 0])
                    # plt.plot(Y_val_new[:24000, 0])
                    # plt.legend(['output', 'y'])
                    # plt.show()
                    # plt.plot(output_strong[:24000, 1])
                    # plt.plot(Y_val_new[:24000, 1])
                    # plt.legend(['output', 'y'])
                    # plt.show()
                    #
                    # plt.plot(output_strong[:24000, 2])
                    # plt.plot(Y_val_new[:24000, 2])
                    # plt.legend(['output', 'y'])
                    # plt.show()
                    #
                    # plt.plot(output_strong[:24000, 3])
                    # plt.plot(Y_val_new[:24000, 3])
                    # plt.legend(['output', 'y'])
                    # plt.show()
                    #
                    # plt.plot(output_strong[:24000, 4])
                    # plt.plot(Y_val_new[:24000, 4])
                    # plt.legend(['output', 'y'])
                    # plt.show()

                    print("STRONG SCORES:")
                    print("Validation")
                    print(multilabel_confusion_matrix(Y_val, output_strong))
                    print(classification_report(Y_val, output_strong))
                    print(hamming_loss(Y_val, output_strong))
                    # print("Test")
                    # print(multilabel_confusion_matrix(Y_test, output_strong_test))
                    # print(classification_report(Y_test, output_strong_test))
                    # print(hamming_loss(Y_test, output_strong_test))
                    print("Test")
                    print(multilabel_confusion_matrix(Y_test_r, output_strong_train))
                    print(classification_report(Y_test_r, output_strong_train))
                    # print(hamming_loss(Y_train, output_strong_train))


                    # ### WEAK SCORES
                    #print("WEAK SCORES:")
                    # print("Test")
                    # print(multilabel_confusion_matrix(Y_test_weak, output_weak_test))
                    # print(classification_report(Y_test_weak, output_weak_test))
                    # print(hamming_loss(Y_test_weak, output_weak_test))
                    # print("Validation")
                    # print(multilabel_confusion_matrix(Y_val_weak, output_weak))
                    # print(classification_report(Y_val_weak, output_weak))
                    # print(hamming_loss(Y_val_weak, output_weak))
                    # print("Train")
                    # print(multilabel_confusion_matrix(Y_train_weak, output_weak_train))
                    # print(classification_report(Y_train_weak, output_weak_train))
                    # print(hamming_loss(Y_train_weak, output_weak_train))