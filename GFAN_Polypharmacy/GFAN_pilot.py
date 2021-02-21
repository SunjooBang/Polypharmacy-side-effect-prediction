



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
Epochs = 100  #@param {type:"slider", min:1000, max:1000000, step:1}
Patience = 100 #@param {type:"slider", min:1, max:500, step:1}
Learning_Rate = 0.001 #@param {type:"slider", min:0, max:0.1, step:0.0001}
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

# Label idx split (train/validation/test split - 80%/10%/10%)
idx_train = list(range(int(len(Poly_Node_Label)*0.8)))
idx_val = list(range(int(len(Poly_Node_Label)*0.8), int(len(Poly_Node_Label)*0.9)))
idx_test = list(range(int(len(Poly_Node_Label)*0.9),len(Poly_Node_Label)))

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

############### GFAN
### model_0
model = GAT(hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
            activation=tf.nn.elu, residual=False)
### save weight_0
#model.save_weights("./training_1/model_0_weight")



####################

delta = np.empty((4, Poly_Node_Feature.shape[0], Poly_Node_Feature.shape[1]))
delta[0, :, :] = np.ones((Poly_Node_Feature.shape[0], Poly_Node_Feature.shape[1]))


delta[1, :, :] = pd.read_csv('./training_1/GFAN_300epoch_delta_1.csv', header=None)
delta[2, :, :] = pd.read_csv('./training_1/GFAN_300epoch_delta_2.csv', header=None)
delta[3, :, :] = pd.read_csv('./training_1/GFAN_300epoch_delta_3.csv', header=None)


train_loss_avg_trend = np.empty((3))  # to plot train_loss per epochs
val_loss_avg_trend = np.empty((3))   # to plot val_loss per epochs

#np.nan_to_num(np.array(train_loss_avg_trend), copy=False)
#print(train_loss_avg_trend)

for i in range(3): #  outepoch = 3 (0,1,2)
    print(i)
    model = GAT(hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
            activation =tf.nn.elu, residual=False)

    features[:, :, :] *= delta[i, :, :]


    ### load weight_0
    model.load_weights("./training_2/model_"+str(i)+"_weight")

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




        print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
              (train_loss_avg / tr_step, train_acc_avg / tr_step,
               val_loss_avg / vl_step, val_acc_avg / vl_step))

#        train_loss_avg_trend += train_loss_avg / tr_step
#        val_loss_avg_trend += val_loss_avg / vl_step

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

        print("outepoch i = ", i, " and inepoch = ", epoch)

### save model_1, model_2, model_3
    model.save_weights("./training_2/model_"+str(i+1)+"_weight")
    print("saved model_", i+1)



###
    Errors_k = np.zeros((Poly_Node_Feature.shape[0], Poly_Node_Feature.shape[1]+1))  # k =0: E_0, k >1: E_k
    for k in range(Poly_Node_Feature.shape[1]+1):  # 0~3648

        temp_data = np.copy(Poly_Node_Feature)
        if k == 0:
            k_0_Poly_Node_Feature= temp_data
        else:
            temp_data[:, k-1] = 0
            k_0_Poly_Node_Feature = temp_data



        k_0_Poly_Node_Feature = k_0_Poly_Node_Feature[np.newaxis]
        k_0_tr_step = 0
        k_0_tr_size = features.shape[0]
        k_0_tr_acc = 0.0
        while k_0_tr_step * batch_size < k_0_tr_size:

            if Sparse:
                bbias = biases
            else:
                bbias = biases[k_0_tr_step * batch_size:(k_0_tr_step + 1) * batch_size]

            out_tr, acc_tr, loss_value_tr = evaluate(model,
                                                inputs=k_0_Poly_Node_Feature[k_0_tr_step * batch_size:(k_0_tr_step + 1) * batch_size],
                                                bias_mat=bbias,
                                                lbl_in=y_train[k_0_tr_step * batch_size:(k_0_tr_step + 1) * batch_size],
                                                msk_in=train_mask[k_0_tr_step * batch_size:(k_0_tr_step + 1) * batch_size],
                                                training=False)

            k_0_tr_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_tr, labels=y_train)  # N x 1
            k_0_tr_loss = tf.reduce_sum(k_0_tr_loss, axis=0)
            k_0_tr_loss = tf.reduce_mean(k_0_tr_loss, axis=1)
            k_0_tr_acc += acc_tr
            k_0_tr_step += 1

        Errors_k[:, k] = k_0_tr_loss.numpy().T # N x F+1

        print("outepoch i = ", i, " and feature  k = ", k)

    # delta
    delta[i+1, :, :] = Errors_k[:, 1:] / Errors_k[:, 0].reshape((-1, 1))  # (N x F) / (N x 1)
    print("delta ", i+1, " was added")

##############################################################################################

np.savetxt('./training_1/GFAN_300epoch_delta_1.csv', delta[1, :, :], delimiter=',')
np.savetxt('./training_1/GFAN_300epoch_delta_2.csv', delta[2, :, :], delimiter=',')
np.savetxt('./training_1/GFAN_300epoch_delta_3.csv', delta[3, :, :], delimiter=',')
np.savetxt('./training_1/GFAN_300epoch_Errors_k.csv', Errors_k, delimiter=',')
np.savetxt('./training_1/GFAN_out_ts.csv', tf.reshape(out_tr, (14247, 1308)), delimiter=',')





#############

# test AUC multilabel case
model = GAT(hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
            activation=tf.nn.elu, residual=False)


model.load_weights("./training_1/model_3_weight")



temp_data = np.copy(Poly_Node_Feature)
Poly_Node_Feature = temp_data

Poly_Node_Feature = Poly_Node_Feature[np.newaxis]
ts_step = 0
ts_size = features.shape[0]
ts_acc = 0.0

while ts_step * batch_size < ts_size:

    if Sparse:
        bbias = biases
    else:
        bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]

    out_ts, acc_ts, loss_value_ts = evaluate(model,
                                             inputs=Poly_Node_Feature[
                                                    ts_step * batch_size:(ts_step + 1) * batch_size],
                                             bias_mat=bbias,
                                             lbl_in=y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             msk_in=test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             training=False)
    ts_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_ts, labels=y_test)  # N x 1
    ts_loss = tf.reduce_sum(ts_loss, axis=0)
    ts_loss = tf.reduce_mean(ts_loss, axis=1)
    ts_acc += acc_ts
    ts_step += 1



######### AUC

from sklearn.metrics import roc_auc_score

ts = tf.keras.activations.sigmoid(out_ts).numpy()
pred_test_temp = tf.reshape(ts, (14247, 1308)).numpy()
pred_test = pred_test_temp[12822:14246, :]  # 12822~14246 test set


y_test___ = y_test[0, 12822:14246, :]
mc_AUC = []
for mc in range(Poly_Node_Label.shape[1]):  # mc=0~1307
    mc_y_test = y_test___[:, mc]
    mc_pred_test = pred_test[:, mc]


    if np.unique(mc_y_test).size != 1:
        mc_AUC_temp = roc_auc_score(mc_y_test, mc_pred_test, average='macro')
        mc_AUC = np.append(mc_AUC, mc_AUC_temp)


    else:
        mc_AUC = np.append(mc_AUC, [0])

np.savetxt('./training_1/mc_AUC.csv', mc_AUC, delimiter=',')


import numpy as np
from sklearn.metrics import average_precision_score


# precision recall curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


#precision = dict()
#recall = dict()
#average_precision = dict()


precision = []
recall = []
average_precision = []

for i in range(1308):
    precision_, recall_, _ = precision_recall_curve(y_test___[:, i],
                                                        pred_test[:, i])
    precision.append(precision_)
    recall.append(recall_)
    average_precision.append(average_precision_score(y_test___[:, i], pred_test[:, i]))

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test___.ravel(), pred_test.ravel())
average_precision["micro"] = average_precision_score(y_test___, pred_test, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

##############






###############

# v_2955_5090 = [CID000002955,CID000005090]: idx 6087, v_1065_5090 = [CID000001065,CID000005090]: idx 955
Poly_Node.loc[(Poly_Node["STITCH 1"] == "CID000002955") & (Poly_Node["STITCH 2"] == "CID000005090")]
Poly_Node.loc[(Poly_Node["STITCH 1"] == "CID000001065") & (Poly_Node["STITCH 2"] == "CID000005090")]
idx_2955_5090 = 6087
idx_1065_5090 = 955
last_delta = pd.read_csv('./training_1/GFAN_300epoch_delta_3.csv')

h_2955_5090 = Poly_Node_Feature.loc[idx_2955_5090, :]
h_1065_5090 = Poly_Node_Feature.loc[idx_1065_5090, :]

delta_2955_5090 = last_delta.loc[idx_2955_5090, :]
delta_1065_5090 = last_delta.loc[idx_1065_5090, :]

np.savetxt('./training_1/h_2955_5090.csv', h_2955_5090, delimiter=',')
np.savetxt('./training_1/h_1065_5090.csv', h_1065_5090, delimiter=',')
np.savetxt('./training_1/delta_2955_5090.csv', delta_2955_5090, delimiter=',')
np.savetxt('./training_1/delta_1065_5090.csv', delta_1065_5090, delimiter=',')

combo = pd.read_csv('./training_1/bio-decagon-combo.csv')
combo.head
combo.columns
sideeffectsname = combo.loc[:, ['Polypharmacy Side Effect', 'Side Effect Name']]
sideeffectsname.drop_duplicates()
sideeffectsname.to_csv(r'./training_1/sideeffectsname.csv', index = False)