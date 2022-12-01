'''
Created on Jul 28, 2017

@author: kashefy
'''
import tensorflow as tf

def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(\
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op