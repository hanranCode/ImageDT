# coding: utf-8
import tensorflow as tf

def smooth_l1_loss(y_true, y_pred):
  delta = 0.5
  x = tf.abs(y_true - y_pred)
  loss = tf.where(x < delta, 0.5 * x ** 2, delta * (x - 0.5 * delta))
  return loss