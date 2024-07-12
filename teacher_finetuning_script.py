import os
from datetime import datetime
import argparse
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from utils_func import *
import random as python_random
import network.CRNN_t
from params import params, uk_params, refit_params
from data.post_processing import *
from network.CRNN_custom import *
from network.metrics_losses import *
from focal_loss import *


parser = argparse.ArgumentParser(description="Knowledge Distillation for Transfer Learning")


parser.add_argument("--gpu", type=str, default="5", help="GPU")
parser.add_argument("--fine_tuning", type=bool, default=True,help="Flag to fine-tune the teacher before KD")
parser.add_argument("--trainable", type=bool, default=False,help="Flag to freeze teacher layers during fine-tuning")
arguments = parser.parse_args()


if __name__ == '__main__':
    # UK-DALE path

    print('Parameter setting:')
    print('GPU',arguments.gpu)
    print('Fine-tuning', arguments.fine_tuning)

    model_ = 'strong_weakUK'  #'strong_weakUK'

    # set seeds for reproducible results
    random.seed(123)
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)
    os.environ['PYTHONHASHSEED'] = str(123)


    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # to disable auto-tuning feature and use deterministi operation
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"]= arguments.gpu #"6"


    # Flag Inizialization
    flag = arguments.fine_tuning
    if flag:
        test = False
    else:
        test= True
    print('Flag', flag)
    strong = False
    strong_weak = True #se avessi voluto escludere da alcuni segmenti le labels weak
    test_unseen = True
    weak_counter = True
    validation_refit = False
    val_only_weak_re = False
    classes = 6


    if model_ == 'strong_weakREFIT':
        X_train_u = np.load('/raid/users/eprincipi/KD_agg_REFIT_pretrain/new_X_train.npy')
        Y_train_u = np.load('/raid/users/eprincipi/KD_labels_REFIT_pretrain/new_Y_train.npy')
        Y_train_weak_u = np.load('/raid/users/eprincipi/KD_labels_REFIT_pretrain/new_Y_train_weak.npy')
        X_val_u = np.load('/raid/users/eprincipi/KD_agg_REFIT_pretrain/new_X_val.npy')
        Y_val_u = np.load('/raid/users/eprincipi/KD_labels_REFIT_pretrain/new_Y_val.npy')
        Y_val_weak_u = np.load('/raid/users/eprincipi/KD_labels_REFIT_pretrain/new_Y_val_weak.npy')
    if model_ == 'strong_weakUK':

        X_train_u = np.load('/raid/users/eprincipi/KD_agg_UKDALE/new_X_train.npy')
        Y_train_u = np.load('/raid/users/eprincipi/KD_labels_UKDALE/new_Y_train.npy')
        Y_train_weak_u = np.load('/raid/users/eprincipi/KD_labels_UKDALE/new_Y_train_weak.npy')
        X_val_u = np.load('/raid/users/eprincipi/KD_agg_UKDALE/new_X_val.npy')
        Y_val_u = np.load('/raid/users/eprincipi/KD_labels_UKDALE/new_Y_val.npy')
        Y_val_weak_u = np.load('/raid/users/eprincipi/KD_labels_UKDALE/new_Y_val_weak.npy')

    X_train_r = np.load('/raid/users/eprincipi/KD_agg_REFIT/new2_X_train.npy')
    Y_train_r = np.negative(np.ones((10481,2550,6))) #np.load('/raid/users/eprincipi/KD_labels_REFIT/new_Y_train.npy')
    Y_train_weak_r = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_train_weak.npy')

    X_test_r = np.load('/raid/users/eprincipi/KD_agg_REFIT/new2_X_test.npy')
    Y_test_r = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_test.npy')
    Y_test_weak_r = np.load('/raid/users/eprincipi/KD_labels_REFIT/new2_Y_test_weak.npy')

    # TODO CONCATENA I DUE VALIDATION
    val_l = round(len(X_test_r) / 100 * 10 /2 )

    X_val_tot = X_val_u
    Y_val =  Y_val_u
    Y_val_weak = Y_val_weak_u
    X_test_r = X_test_r[val_l:-val_l]
    Y_test_r = Y_test_r[val_l:-val_l]
    Y_test_weak_r = Y_test_weak_r[val_l:-val_l]

    assert(len(X_val_u)==len(Y_val_u))
    assert(len(Y_val_u)==len(Y_val_weak_u))

    x_train = X_train_r
    y_strong_train = Y_train_r
    y_weak_train = Y_train_weak_r

    weak_count(y_weak_train, classes=classes)

    # Standardization with uk-dale values
    if model_ == 'solo_weakUK' or model_ == 'strong_weakUK' or model_=='mixed':
        train_mean = uk_params['mean']
        train_std =  uk_params['std']
    else:
        if model_=='solo_strongUK':
            train_mean = 273.93
            train_std = 382.70
        else:
            train_mean = refit_params['mean']
            train_std = refit_params['std']

    print("Mean train")
    print(train_mean)
    print("Std train")
    print(train_std)
    Y_val = np.where(Y_val>0.5, 1, 0 )
    Y_test_r =np.where(Y_test_r>0.5, 1, 0 )
    x_train = standardize_data(x_train,train_mean, train_std)
    X_val = standardize_data(X_val_tot, train_mean, train_std)
    X_test = standardize_data(X_test_r,train_mean, train_std)
    print(X_val.shape)
    batch_size = 64
    window_size = 2550
    drop = params[model_]['drop']
    kernel = params[model_]['kernel']
    num_layers = params[model_]['layers']
    gru_units = params[model_]['GRU']
    cs = params[model_]['cs']
    only_strong = params[model_]['no_weak']
    temperature = 1
    patience = 30
    type_ = model_
    print(type_)
    lr = 0.002
    weight= 1
    classes = 6
    gamma = K.variable(1.0)
    weight_dyn = WeightAdjuster(weights=gamma)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_F1_score', mode='max', patience=patience, restore_best_weights=True)

    # TEACHER FINE-TUNING
    log_dir_ =  datetime.now().strftime(
        "%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    # Carico il pre-trained model
    pre_trained = params[model_]['pre_trained']
    teacher = network.CRNN_t.CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                             path=pre_trained, only_strong=only_strong, temperature=temperature)

    teacher.summary()
    MODEL = CRNN_custom(teacher)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    MODEL.compile(student_optimizer=optimizer, teacher_optimizer=optimizer,
                  loss={"strong_loss": binary_crossentropy,
                        "weak_loss": BinaryFocalLoss(gamma=0.2)},
                  loss_weights=[gamma], Temperature=1, F1_score=StatefullF1())
    if not test:

        history = MODEL.fit(x=x_train, y=[y_strong_train, y_strong_train, y_weak_train], shuffle=True, epochs=1000,
                            batch_size=batch_size,
                            validation_data=(X_val, [Y_val, Y_val, Y_val_weak]),
                            callbacks=[early_stop, tensorboard], verbose=1)


        MODEL.teacher.save_weights(
                    'teacher_' + type_ + str(weight) + '_' + str(batch_size) + '.h5')

    else:
        MODEL.teacher.load_weights('fine_tuned_teachers/teacher_' + type_ + str(weight) + '_' + str(batch_size) + '.h5')

    soft_teach_strong, output_teacher_test_o, output_teacher_weak_test = teacher.predict(x=X_test)
    soft_val_teach_strong, output_teacher_val, output_teacher_weak_val = teacher.predict(x=X_val)

    shape_test = output_teacher_test_o.shape[0] * output_teacher_test_o.shape[1]
    shape = output_teacher_val.shape[0] * output_teacher_val.shape[1]

    Y_val = Y_val.reshape(shape, classes)
    Y_test = Y_test_r.reshape(shape_test, classes)

    output_strong_test_teach = output_teacher_test_o.reshape(shape_test, classes)
    output_teacher_val = output_teacher_val.reshape(shape, classes)

    thres_strong_teach = thres_analysis(Y_val, output_teacher_val, classes)

    print("Estimated best thresholds teacher:", thres_strong_teach)

    output_teacher_val = app_binarization_strong(output_teacher_val, thres_strong_teach, classes)

    print("STRONG SCORES:")
    print("Teacher Validation")
    b = classification_report(Y_val, output_teacher_val)
    print(b)
    for i in range(classes):

        tn, fp, fn, tp = confusion_matrix(Y_val[:,i], output_teacher_val[:,i]).ravel()
        print(tn / (tn + fp))


