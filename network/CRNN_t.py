import tensorflow as tf
from tensorflow.keras import backend as K
from network.pooling_layer import LinSoftmaxPooling1D
from network.losses import binary_crossentropy,binary_crossentropy_weak
from network.metrics_losses import  StatefullF1


def CRNN_block(x, kernel,drop_out,filters):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel, 1), strides=(1, 1), padding='same',
                                    kernel_initializer='glorot_uniform')(x)
    print("conv_1")
    print(conv_1.shape)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1)
    act_1 = tf.keras.layers.Activation('relu')(batch_norm_1)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(act_1)
    drop_1 = tf.keras.layers.Dropout(drop_out)(pool_1)
    print("drop_1")
    print(drop_1.shape)
    return drop_1


def CRNN_construction(window_size, weight, lr=0.0, classes=0, drop_out = 0.1, kernel = 1, num_layers=1, gru_units=1, cs=False, path = '', only_strong=False, temperature=1.0):

    input_data = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Reshape((window_size,1,1))(input_data)

    for i in range(num_layers):
        filters = 2 ** (i+5)
        CRNN = CRNN_block(x, kernel=kernel, drop_out=drop_out, filters=filters)
        x = CRNN

    spec_x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    print("Reshape")
    print(spec_x.shape)
    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,return_sequences=True))(spec_x)
    print("bi direct")
    print(bi_direct.shape)
    instance_level = tf.keras.layers.Dense(units=classes)(bi_direct)  # activation='sigmoid',
    print("Instance Level")
    print(instance_level.shape)
    frame_l = tf.keras.layers.Lambda(lambda x: x * 1 / temperature)(instance_level)
    strong_level_soft = tf.keras.layers.Activation('sigmoid', name='strong_level_soft')(frame_l)
    frame_level = tf.keras.layers.Activation('sigmoid', name="strong_level")(instance_level)
    pool_bag = LinSoftmaxPooling1D(axis=1)(frame_level)
    bag_level = tf.keras.layers.Activation('sigmoid', name="weak_level")(pool_bag)

    # if cs:
    #         frame_level_final = tf.keras.layers.Multiply(name="strong_level_final")([bag_level, frame_level])
    #         frame_level_final_soft = tf.keras.layers.Multiply(name="strong_level_final_soft")([bag_level, strong_level_soft])
    #         print(frame_level_final.shape)
    #
    #         model_CRNN = tf.keras.Model(inputs=input_data, outputs=[frame_level_final_soft, frame_level_final, bag_level],
    #                                     name="CRNN")
    #         model_CRNN.load_weights(path)
    #
    #         conv_1 = model_CRNN.get_layer('conv2d')
    #         conv_1.trainable = False
    #         conv2d_1 = model_CRNN.get_layer('conv2d_1')
    #         conv2d_1.trainable = False
    #         conv2d_2 = model_CRNN.get_layer('conv2d_2')
    #         conv2d_2.trainable = False
    #         bidirectional = model_CRNN.get_layer('bidirectional')
    #         bidirectional.trainable = False
    #
    #
    #         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #
    #         # model_CRNN.compile(optimizer=optimizer, loss={
    #         #     "strong_level_final_soft": binary_crossentropy,
    #         #     "strong_level_final": binary_crossentropy,
    #         #     "weak_level": binary_crossentropy_weak,
    #         # }, metrics=StatefullF1(), loss_weights=[1,1, weight])



    model_CRNN = tf.keras.Model(inputs=input_data, outputs=[strong_level_soft, frame_level, bag_level], name="CRNN")
    model_CRNN.load_weights(path)
    conv_1 = model_CRNN.get_layer('conv2d')
    conv_1.trainable = True
    conv2d_1 = model_CRNN.get_layer('conv2d_1')
    conv2d_1.trainable = True
    conv2d_2 = model_CRNN.get_layer('conv2d_2')
    conv2d_2.trainable = True
    bidirectional = model_CRNN.get_layer('bidirectional')
    bidirectional.trainable = True

    return model_CRNN

