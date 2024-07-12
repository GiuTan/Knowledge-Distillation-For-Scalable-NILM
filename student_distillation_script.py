import os
from datetime import datetime
import argparse
from tensorflow.keras.callbacks import TensorBoard
import network.CRNN_t
import network.CRNN
from utils_func import *
from params import *
import json
import random as python_random
from network.TEACH_STU import *
import network.CRNN_t
import tensorflow as tf
from metrics import *
from params import params, uk_params, refit_params
from data.post_processing import *
from network.CRNN_custom import *
from network.metrics_losses import *



parser = argparse.ArgumentParser(description="Knowledge Distillation for Transfer Learning")


parser.add_argument("--gpu", type=str, default="2", help="GPU")
parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for KD")
parser.add_argument("--beta", type=float, default=0.7, help="KD loss weight")
parser.add_argument("--fine_tuning", type=bool, default=False,help="Flag to fine-tune the teacher before KD")
parser.add_argument("--model", type=str, default="strong_weakREFIT", help="UKDALE or REFIT pre-training selection")
parser.add_argument("--num_conv", type=int, default=1, help="Number of convolutional blocks")
parser.add_argument("--num_GRUnits", type=int, default=64, help="Number of GRUnits")
arguments = parser.parse_args()


if __name__ == '__main__':
    # UK-DALE path

    print('Parameter setting:')
    print('GPU',arguments.gpu)
    print('Temperature', arguments.temperature)
    print('Beta', arguments.beta)
    print('Fine-tuning', arguments.fine_tuning)
    print('Model', arguments.model)
    print('Number of convolutional blocks', arguments.num_conv)
    print('Number of gated recurrent units', arguments.num_GRUnits)

    print('Start')

    # set seeds for reproducible results
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # to disable auto-tuning feature and use deterministic operation
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    os.environ['PYTHONHASHSEED'] = str(123)

    os.environ["CUDA_VISIBLE_DEVICES"]= arguments.gpu

    model_ = arguments.model


    # Flag Inizialization
    flag = arguments.fine_tuning
    if flag:
        test = False
    else:
        test= True
    print('Flag', flag)

    # LOADING DATA FROM .JSON FOR LABELS STRONG AND WEAK AND .NPY FOR AGGREGATE
    if model_ == 'strong_weakREFIT':
        X_train_u = np.load('')
        X_val_u = np.load('')
        Y_val_u = np.load('')
        Y_val_weak_u = np.load('')
    if model_ == 'strong_weakUK':
        X_train_u = np.load('')
        X_val_u = np.load('')
        Y_val_u = np.load('')
        Y_val_weak_u = np.load('')

    X_train_r = np.load('')
    Y_train_r = np.negative(np.ones((len(X_train_r),window_size,classes)))
    Y_train_weak_r = np.load('')
    X_test_r = np.load('')
    Y_test_r = np.load('')
    Y_test_weak_r = np.load('')


    # TODO CONCATENA I DUE VALIDATION
    val_l = round(len(X_test_r) / 100 * 10 / 2)

    X_val_tot = X_val_u
    Y_val = Y_val_u
    Y_val_weak =  Y_val_weak_u
    X_test_r = X_test_r[val_l:-val_l]
    Y_test_r = Y_test_r[val_l:-val_l]
    Y_test_weak_r = Y_test_weak_r[val_l:-val_l]

    Y_val = np.where(Y_val >= 0.5, 1, Y_val)
    Y_val = np.where((Y_val != -1) & (Y_val < 0.5), 0, Y_val)
    Y_test = np.where(Y_test_r >= 0.5, 1, Y_test_r)
    Y_test = np.where((Y_test != -1) & (Y_test < 0.5), 0, Y_test)


    assert(len(X_val_u)==len(Y_val_u))
    assert(len(Y_val_u)==len(Y_val_weak_u))

    x_train = X_train_r
    y_strong_train = Y_train_r
    y_weak_train = Y_train_weak_r


    # Standardization with uk-dale values
    if model_ == 'strong_weakUK':
        train_mean = uk_params['mean']
        train_std =  uk_params['std']
    else:
        if model_ == 'strong_weakREFIT':
            train_mean = refit_params['mean']
            train_std = refit_params['std']

    print("Mean train")
    print(train_mean)
    print("Std train")
    print(train_std)

    x_train = standardize_data(x_train,train_mean, train_std)
    X_val = standardize_data(X_val_tot, train_mean, train_std)
    X_test = standardize_data(X_test_r,train_mean, train_std)
    print(X_val.shape)

    drop = params[model_]['drop']
    kernel = params[model_]['kernel']
    num_layers = params[model_]['layers']
    gru_units = params[model_]['GRU']
    cs = params[model_]['cs']
    only_strong = params[model_]['no_weak']
    temperature = arguments.temperature
    bet = arguments.beta
    pat = 30
    if arguments.num_conv < 3 or arguments.num_GRUnits < 64:
        type_ = model_ + '_T' + str(temperature) + '_' + str(bet) + 'KD_' + '_6classes_new22_REDUCED_' + str(arguments.num_conv) +'_'+ str(arguments.num_GRUnits)
    else:
        type_ =  model_ + '_T'+ str(temperature) + '_'+str(bet)+'KD_' + '_6classes_new22'
    print(type_)
    lr = 0.002
    weight= 1
    classes = 6
    gamma = K.variable(1.0)
    weight_dyn = WeightAdjuster(weights=gamma)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_F1_score', mode='max', patience=15, restore_best_weights=True)

    # TEACHER FINE-TUNING
    log_dir_ = datetime.now().strftime(
        "%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    print('Load fine-tuned Teacher')

    if model_ == 'strong_weakREFIT':
        fine_tuned = 'finetuned_teacher_REFIT.h5'
    else:
        fine_tuned = 'finetuned_teacher_UKDALE.h5'

    teacher = network.CRNN_t.CRNN_construction(window_size, weight, lr=lr, classes=classes, drop_out=drop,
                                               kernel=kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                                               path=fine_tuned, only_strong=only_strong, temperature=temperature)

    MODEL = CRNN_custom(teacher)
    student = network.CRNN.CRNN_construction(window_size,weight, lr=lr, classes=classes, drop_out=drop, kernel = kernel, num_layers=arguments.num_conv, gru_units=arguments.num_GRUnits, cs=cs)
    student.summary()

    # CREAZIONE CALLBACKS

    csv_logger = tf.keras.callbacks.CSVLogger( type_+ '.csv')
    alpha = 1 - bet
    beta = bet
    gamma = K.variable(1.0)
    weight_dyn_TS = WeightAdjuster_TS(weights=gamma)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_stu',  mode='max',
                                                  patience=pat, restore_best_weights=True)

    log_dir_ = datetime.now().strftime(
        "%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    # CREAZIONE MODELLO TEACHER-STUDENT

    MODEL_ = STU_TEACH(student, MODEL.teacher)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    MODEL_.compile(student_optimizer=optimizer, teacher_optimizer=optimizer,
                  loss={"student_loss": loss_fn_sup, "KD_loss": loss_fn_sup},
                  loss_weights=[alpha, beta, gamma], Temperature=temperature, F1_score=StatefullF1())
    history_ = MODEL_.fit(x=x_train, y=[y_strong_train,y_strong_train, y_weak_train], shuffle=True, epochs=epochs,validation_data=(X_val, [Y_val,Y_val, Y_val_weak]), callbacks=[early_stop, tensorboard, csv_logger,weight_dyn_TS], batch_size=batch_size, verbose=1) #
    MODEL_.student.save_weights('student'+type_ +'.h5')
    print('MODEL SAVED!')
    print('START COMPUTING PREDICTIONS')
    val_soft_strong , output_strong, output_weak = MODEL_.predict(x=X_val)
    test_soft_strong, output_strong_test_o, output_weak_test = MODEL_.predict(x=X_test)
    soft_teach_strong, output_teacher_test_o, output_teacher_weak_test = MODEL.teacher.predict(x=X_test)
    soft_val_teach_strong,output_teacher_val, output_teacher_weak_val = MODEL.teacher.predict(x=X_val)

    print(Y_val.shape)
    print(output_strong.shape)

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]


    Y_val = Y_val.reshape(shape, classes)
    Y_test = Y_test.reshape(shape_test, classes)

    output_strong = output_strong.reshape(shape, classes)
    output_strong_test = output_strong_test_o.reshape(shape_test, classes)
    output_strong_test_teach = output_teacher_test_o.reshape(shape_test, classes)

    output_teacher_val = output_teacher_val.reshape(shape, classes)

    print('OPTIMAL THRESHOLD ESTIMATION ON VALIDATION SET')
    thres_strong_stu = thres_analysis(Y_val, output_strong,classes)
    thres_strong_teach = thres_analysis(Y_val, output_teacher_val, classes)

    output_weak_test = output_weak_test.reshape(output_weak_test.shape[0] * output_weak_test.shape[1], classes)
    output_weak = output_weak.reshape(output_weak.shape[0] * output_weak.shape[1], classes)
    thres_weak = [0.501, 0.501, 0.501, 0.501, 0.501]

    assert (Y_val.shape == output_strong.shape)

    print("Estimated best thresholds for the student:", thres_strong_stu)
    print("Estimated best thresholds fro the teacher:", thres_strong_teach)

    output_strong_test = app_binarization_strong(output_strong_test, thres_strong_stu, classes)
    output_strong_test_teach = app_binarization_strong(output_strong_test_teach, thres_strong_teach, classes)
    #
    output_strong = app_binarization_strong(output_strong, thres_strong_stu, classes)


    print("STRONG SCORES:")
    print("Student Validation")
    b = classification_report(Y_val, output_strong)
    print(b)
    print("Student Test")
    a = classification_report(Y_test, output_strong_test)
    print(a)
    print("Teacher Test")
    c = classification_report(Y_test, output_strong_test_teach)
    print(c)

    with open(type_ +
              'scores.txt',
              'a+') as f:
        print('classification report validation: %s' % b, file=f)
        print('classification report test student: %s' % a, file=f)
        print('classification report test teacher: %s' % c, file=f)
        print('', file=f)
