import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras import backend as K


def binary_crossentropy(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true,new_pred)
    return tf.reduce_mean(loss)

def binary_crossentropy_weak(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true, new_pred)

    return tf.reduce_mean(loss)

def f1_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    y_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    # tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    # fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    # fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    #
    # p = tp / (tp + fp + K.epsilon())
    # r = tp / (tp + fn + K.epsilon())
    #
    # f1 = 2 * p * r / (p + r + K.epsilon())
    # f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    # return 1 - K.mean(f1)

    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    tn = tf.reduce_sum((1 - y_pred) * (1 - y_true), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

def custom_f1_score(y_true, y_pred):
    y_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    y_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    def recall_m(y_true, y_pred):
         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
         Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

         recall = TP / (Positives + K.epsilon())
         return recall

    def precision_m(y_true, y_pred):
         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
         Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

         precision = TP / (Pred_Positives + K.epsilon())
         return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class StatefullF1(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', n_class=6, dim=2550, average='macro', epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super(StatefullF1, self).__init__(name=name, **kwargs)

        # initializing state variables
        self.tp = self.add_weight(name='tp', shape=(dim,n_class), initializer='zeros')  # initializing true positives
        self.actual_positives = self.add_weight(name='ap', shape=(dim,n_class),
                                                initializer='zeros')  # initializing actual positives
        self.predicted_positives = self.add_weight(name='pp', shape=(dim,n_class),
                                                   initializer='zeros')  # initializing predicted positives

        # initializing other atrributes that wouldn't be changed for every object of this class

        self.n_class = n_class
        self.average = average
        self.epsilon = epsilon
        self.dim = dim

    def update_state(self, ytrue, ypred, sample_weight=None):

        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)
        ytrue = tf.multiply(ytrue, tf.cast(tf.not_equal(ytrue, -1), tf.float32))
        ypred = tf.multiply(ypred, tf.cast(tf.not_equal(ytrue, -1), tf.float32))

        self.tp.assign_add(tf.reduce_sum(tf.clip_by_value(ytrue * ypred,0,1) , axis=0))  # updating true positives attribute
        self.predicted_positives.assign_add(tf.reduce_sum(tf.clip_by_value(ypred,0,1) , axis=0))  # updating predicted positives attribute
        self.actual_positives.assign_add(tf.reduce_sum(tf.clip_by_value(ytrue,0,1) , axis=0))  # updating actual positives attribute

    def result(self):
        self.precision = self.tp / (self.predicted_positives + self.epsilon)  # calculates precision
        self.recall = self.tp / (self.actual_positives + self.epsilon)  # calculates recall

        # calculating fbeta score
        self.fb = (2) * self.precision * self.recall / ( self.precision + self.recall + self.epsilon)

        if self.average == 'weighted':
            return tf.reduce_sum(self.fb * self.actual_positives / tf.reduce_sum(self.actual_positives))

        elif self.average == 'raw':
            return self.fb

        return tf.reduce_mean(self.fb)

    def reset_states(self):
        self.tp.assign(tf.cast(tf.zeros((self.dim,self.n_class)),tf.float32))  # resets true positives to zero
        self.predicted_positives.assign(tf.cast(tf.zeros((self.dim,self.n_class)),tf.float32))  # resets predicted positives to zero
        self.actual_positives.assign(tf.cast(tf.zeros((self.dim,self.n_class)),tf.float32))  # resets actual positives to zero