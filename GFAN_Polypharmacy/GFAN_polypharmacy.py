
from __future__ import absolute_import, print_function, division, unicode_literals
import pandas as pd
import numpy as np
import pickle
import dill
df = pd.DataFrame


## data import
Poly_Node = pd.read_csv ('./Preprocessed_data/Poly_Node.csv')
Poly_Node_Label = pd.read_csv ('./Preprocessed_data/Poly_Node_Label.csv')
Poly_Node_Feature = pd.read_csv ('./Preprocessed_data/Poly_Node_Feature.csv')
Poly_Edge_list = pd.read_csv ('./Preprocessed_data/Poly_Edge_list.csv')


# Read data
#now_benchmark = 'Poly'
#Data_file = Poly_Node_Feature[] # feature
#Adj_file = Poly_Edge_list # adj list not matrix
#Y_file = Poly_Node_Label # train label

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

print(tf.__version__)
print(tf.executing_eagerly())

Dataset = 'Poly'  #param ["Poly", ...]
Sparse = False #@param {type:"boolean"}
Batch_Size = 1 #@param {type:"slider", min:1, max:1000, step:1}
Epochs = 300  #@param {type:"slider", min:1000, max:1000000, step:1}
Patience = 100 #@param {type:"slider", min:1, max:500, step:1}
Learning_Rate = 0.005 #@param {type:"slider", min:0, max:0.1, step:0.0001}
Weight_Decay = 0.0005 #@param {type:"slider", min:0, max:0.1, step:0.0001}
ffd_drop = 0.6 #@param {type:"slider", min:0, max:1, step:0.01}
attn_drop = 0.6 #@param {type:"slider", min:0, max:1, step:0.01}
Residual = False #@param {type:"boolean"}


dataset = Dataset

# training params
batch_size = Batch_Size
nb_epochs = Epochs
patience = Patience
lr = Learning_Rate
l2_coef = Weight_Decay
residual = Residual

hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer

nonlinearity = tf.nn.elu
optimizer = tf.keras.optimizers.Adam(lr = lr)


import tensorflow as tf
import numpy as np

class GAT(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(GAT, self).__init__()
        '''
        hid_units: This is the number of hidden units per each attention head in each layer (8). Array of hidden layer dimensions
        n_heads: This is the additional entry of the output layer [8,1]. More specifically the output that calculates attn    
        nb_classes: This refers to the number of classes (7)
        nb_nodes: This refers to the number of nodes (2708)    
        activation: This is the activation function tf.nn.elu
        residual: This determines whether we add seq to ret (False)
        '''
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.inferencing = inference(n_heads, hid_units, nb_classes, nb_nodes, Sparse=Sparse, ffd_drop=ffd_drop,
                                     attn_drop=attn_drop, activation=activation, residual=residual)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(self, logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)

        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure

    def __call__(self, inputs, training, bias_mat, lbl_in, msk_in):
        logits = self.inferencing(inputs=inputs, bias_mat=bias_mat, training=training)

        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])

        loss = self.masked_sigmoid_cross_entropy(log_resh, lab_resh, msk_resh)

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        loss = loss + lossL2
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, accuracy, loss


def train(model, inputs, bias_mat, lbl_in, msk_in, training):
    with tf.GradientTape() as tape:
        logits, accuracy, loss = model(inputs=inputs,
                                       training=True,
                                       bias_mat=bias_mat,
                                       lbl_in=lbl_in,
                                       msk_in=msk_in)

    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)

    return logits, accuracy, loss


def evaluate(model, inputs, bias_mat, lbl_in, msk_in, training):
    logits, accuracy, loss = model(inputs=inputs,
                                   bias_mat=bias_mat,
                                   lbl_in=lbl_in,
                                   msk_in=msk_in,
                                   training=False)
    return logits, accuracy, loss


import time
import numpy as np
import tensorflow as tf
import networkx as nx
from GFAN_Polypharmacy.utils import *
from GFAN_Polypharmacy.attn_layers import *

# from models import GAT
# from utils import process


print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))


# Adjacency matrix
G = nx.Graph()
G.add_edges_from(np.array(Poly_Edge_list))
adj = nx.adjacency_matrix(G)

# Label idx split (train/validation/test split - 40%/30%/30%)
idx_train = list(range(int(len(Poly_Node_Label)*0.4)))
idx_val = list(range(int(len(Poly_Node_Label)*0.4), int(len(Poly_Node_Label)*0.7)))
idx_test = list(range(int(len(Poly_Node_Label)*0.7),len(Poly_Node_Label)))

train_mask = sample_mask(idx_train, Poly_Node_Label.shape[0])
val_mask = sample_mask(idx_val, Poly_Node_Label.shape[0])
test_mask = sample_mask(idx_test, Poly_Node_Label.shape[0])

y_train = np.zeros(Poly_Node_Label.shape)
y_val = np.zeros(Poly_Node_Label.shape)
y_test = np.zeros(Poly_Node_Label.shape)
y_train[train_mask, :] = Poly_Node_Label.iloc[train_mask, :]
y_val[val_mask, :] = Poly_Node_Label.iloc[val_mask, :]
y_test[test_mask, :] = Poly_Node_Label.iloc[test_mask, :]
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

# if you need normalization!!
#features, spars = preprocess_features(features)

features = np.array(Poly_Node_Feature, dtype=np.float32)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

print(f'These are the parameters')
print(f'batch_size: {batch_size}')
print(f'nb_nodes: {nb_nodes}')
print(f'ft_size: {ft_size}')
print(f'nb_classes: {nb_classes}')

if Sparse:
    biases = preprocess_adj_bias(adj)

else:
    adj = adj.todense()
    adj = adj[np.newaxis]
    biases = adj_to_bias(adj, [nb_nodes], nhood=1)
    biases = biases.astype(np.float32)

model = GAT(hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
            activation=tf.nn.elu, residual=False)
#dill.dump(model, file = open("./training_1/model.pickle", "wb"))
## model save

#for i in range(3):

#    model = dill.load(open("./training_1/model.pickle", "rb"))

print('model: ' + str('SpGAT' if Sparse else 'GAT'))

vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0

train_loss_avg = 0
train_acc_avg = 0
val_loss_avg = 0
val_acc_avg = 0

model_number = 0


for epoch in range(nb_epochs):
    ###Training Segment###
    tr_step = 0
    tr_size = features.shape[0]
    while tr_step * batch_size < tr_size:

        if Sparse:
            bbias = biases
        else:
            bbias = biases[tr_step * batch_size:(tr_step + 1) * batch_size]

        _, acc_tr, loss_value_tr = train(model,
                                         inputs=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                         bias_mat=bbias,
                                         lbl_in=y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                                         msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                                         training=True)
        train_loss_avg += loss_value_tr
        train_acc_avg += acc_tr
        tr_step += 1

    ###Validation Segment###
    vl_step = 0
    vl_size = features.shape[0]
    while vl_step * batch_size < vl_size:

        if Sparse:
            bbias = biases
        else:
            bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]

        _, acc_vl, loss_value_vl = evaluate(model,
                                            inputs=features[vl_step * batch_size:(vl_step + 1) * batch_size],
                                            bias_mat=bbias,
                                            lbl_in=y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                                            msk_in=val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                                            training=False)
        val_loss_avg += loss_value_vl
        val_acc_avg += acc_vl
        vl_step += 1

    print('%d - Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
          (epoch+1, train_loss_avg / tr_step, train_acc_avg / tr_step,
           val_loss_avg / vl_step, val_acc_avg / vl_step))

    ###Early Stopping Segment###

    if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
        if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
            vacc_early_model = val_acc_avg / vl_step
            vlss_early_model = val_loss_avg / vl_step
            working_weights = model.get_weights()
        vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
        vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == patience:
            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
            print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
            model.set_weights(working_weights)
            break

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0

dill.dump(model, file = open("./training_1/model.pickle", "wb"))


###Testing Segment### Outside of the epochs

ts_step = 0
ts_size = features.shape[0]
ts_loss = 0.0
ts_acc = 0.0
while ts_step * batch_size < ts_size:

    if Sparse:
        bbias = biases
    else:
        bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]

    out_ts, acc_ts, loss_value_ts = evaluate(model,
                                        inputs=features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                        bias_mat=bbias,
                                        lbl_in=y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                        msk_in=test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                        training=False)
    ts_loss += loss_value_ts
    ts_acc += acc_ts
    ts_step += 1

print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)
# print('Test loss: %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
#                  (train_loss_avg/tr_step, train_acc_avg/tr_step,
#                  val_loss_avg/vl_step, val_acc_avg/vl_step))

model.save_weights("./training_1/model_GAT_weight")