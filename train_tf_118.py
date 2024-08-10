import numpy as np
np.random.seed(1337)
import math
import time
import h5py
import tensorflow as tf
SEED = 2
tf.random.set_seed(seed=SEED)
from nn import ttednn_keras_1
# from spektral.layers import GraphConv
from spektral.utils import batch_iterator
from tensorflow.keras.metrics import (BinaryAccuracy, BinaryCrossentropy, TruePositives, CategoricalAccuracy, CategoricalCrossentropy, Accuracy)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, auc, roc_curve
from utils118 import (load_data, load_para)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	# 使用第 0 号 GPU
# tf.compat.v1.Session()
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
#####################  set parameters  ####################

N             = 118                    # number of node
omega_s       = 100 * math.pi
theta         = math.pi
exp_num       = 1
early_stop    = True
interval      = False
relative      = False
normalize     = False
standard      = False
mode          = 1
move          = False
WSZ           = 11

n_critical    = 0  # thresholds to change loss function
weight        = False
n             = 0  # current epochs

if interval:
    timelength = 61
else:
    timelength = 101

net = 'ttednn_keras_1'
data_set = 'one'
label         = 'freq'
adj_mode      = 2                       # adjacency matrix mode: 1、adj=Y
                                        #                        2、adj=diag(P)+Y
                                        #                        3、adj=P'+Y',P'=P·(1+ω_0/ω_s),Y'=Y_ij·sin(θ_i-θ_j)
chosedlength  =  61                     # length used to train
data_number   =  50                    # same in data/swing_equation/IEEE.py   每组接地中随机初始化个数
TEST_SIZE     =  0.2                    # train:val_test = 6:2:2

F             = chosedlength
if label == 'both':
    n_out     = 4
else:
    n_out     = 1
l2_reg_gcn    = 5e-4
learning_rate = 5e-4                    # Learning rate for Adam
BATCH_SIZE    = 128                     # Batch size
epochs        = 200                       # Number of training epochs
patience      = 100                     # Patience for early stopping

print('choslength : %s \n interval  : %s \n normalize : %s \n standard  : %s \n relative  : %s \n mode      : %s \n move      : %s'
    % (chosedlength, interval, normalize, standard, relative, mode, move)
)

#####################  load data & processing  ####################

X_train, X_val, X_test, Y_train, Y_val, Y_test, Adj_train, Adj_val, Adj_test = load_data(
    N=N, adj_mode= adj_mode, mac_number=54, length=data_number, init=0, T=chosedlength, TEST_SIZE=TEST_SIZE
    , label=label
)
if label != 'both':
    Y_train = Y_train.reshape(-1, 1).astype('float32')
    Y_val   = Y_val.reshape(-1, 1).astype('float32')
    Y_test  = Y_test.reshape(-1, 1).astype('float32')
else:
    pass
print(X_train.shape, Y_train.shape)
#####################  Network setup  ####################

model = ttednn_keras_1(
    N=N,
    T=chosedlength,
    n_out=n_out,
    l2_reg_gcn=l2_reg_gcn
    # ,filters_tcn=16
)

optimizer = Adam(lr=learning_rate)

if label != 'both':
    loss_object   = tf.keras.losses.BinaryCrossentropy()  # for training
    train_loss_fn = BinaryCrossentropy()                  # for model evaluate
    train_acc_fn  = BinaryAccuracy()
    val_loss_fn   = BinaryCrossentropy()
    val_acc_fn    = BinaryAccuracy()
    test_loss_fn  = BinaryCrossentropy()
    test_acc_fn   = BinaryAccuracy()
    acc_fn_1      = TruePositives()

else:
    loss_object   = tf.keras.losses.CategoricalCrossentropy()  # for training
    train_loss_fn = CategoricalCrossentropy()                  # for model evaluate
    train_acc_fn  = CategoricalAccuracy()
    val_loss_fn   = CategoricalCrossentropy()
    val_acc_fn    = CategoricalAccuracy()
    test_loss_fn  = CategoricalCrossentropy()
    test_acc_fn   = CategoricalAccuracy()
    acc_fn_1      = TruePositives()

#####################  Functions  ####################


# Training step
# @tf.function
def train_weight(x, fltr, y, pos_weight=None):
    """
    for class_balanced_loss
    """
    if weight:
        # a = sum(y)
        # b = len(y)
        # if a > 0:
        #     pos_weight = y * b / a + np.ones_like(y) - y
        # else:
        #     pos_weight = np.ones_like(y)
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_object(y, predictions, sample_weight=pos_weight)
            loss += sum(model.losses)
    else:
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_object(y, predictions)
            loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # return loss, acc
    train_loss_fn(y, predictions)
    train_acc_fn(y, predictions)


# Evaluation step
# @tf.function
def evaluate_weight(x, fltr, y, pos_weight=None):
    if weight:
        # a = sum(y)
        # b = len(y)
        # if a > 0:
        #     pos_weight = y * b / a + np.ones_like(y) - y
        # else:
        #     pos_weight = np.ones_like(y)
        predictions = model([x, fltr], training=False)
        val_loss = val_loss_fn(y, predictions, sample_weight=pos_weight)
    else:
        predictions = model([x, fltr], training=False)
        val_loss = val_loss_fn(y, predictions)
    val_loss += sum(model.losses)
    val_acc = val_acc_fn(y, predictions)
    val_loss_fn.reset_states()
    val_acc_fn.reset_states()
    return val_loss, val_acc


# Testing step
@tf.function
def test_weight(x, fltr, y):
    predictions = model([x, fltr], training=False)
    te_loss = test_loss_fn(y, predictions)
    te_loss += sum(model.losses)
    te_acc = test_acc_fn(y, predictions)
    test_loss_fn.reset_states()
    test_acc_fn.reset_states()
    return te_loss, te_acc, predictions

# X_train, X_val, X_test = np.squeeze(X_train[:, :, :, 2]), np.squeeze(X_val[:, :, :, 2]), np.squeeze(X_test[:, :, :, 2])


# Setup training
best_val_loss = 99999
current_patience = patience
curent_batch = 0
batches_in_epoch = int(np.ceil(X_train.shape[0] / BATCH_SIZE))
batches_tr = batch_iterator([X_train, Adj_train, Y_train], batch_size=BATCH_SIZE, epochs=epochs)

# Training loop
loss_train = []
acc_train  = []
loss_val   = []
acc_val    = []
loss_test  = []
acc_test   = []
n = 0
print('\nTraining ------------')
start = time.perf_counter()
if n_critical == 0:
    weight = True
    print('Loss function=Weight BCE')
else:
    print('Loss function=BCE')
    pass
loss_1, acc_1 = evaluate_weight(x=X_train, fltr=Adj_train, y=Y_train)
loss_2, acc_2 = evaluate_weight(x=X_val, fltr=Adj_val, y=Y_val)
loss_3, acc_3 = evaluate_weight(x=X_test, fltr=Adj_test, y=Y_test)
print(
    'Epochs: {:.0f} | ' 'Train loss: {:.4f}, acc: {:.4f} | ' 'Valid loss: {:.4f}, acc: {:.4f} | ' 'Test loss: {:.4f}, acc: {:.4f}'
    .format(n, loss_1, acc_1, loss_2, acc_2, loss_3, acc_3)
)

for batch in batches_tr:

    if n == n_critical and curent_batch == 0:
        weight = True
        print('Loss function=Weight BCE')
    else:
        pass
    curent_batch += 1
    if weight:

        if n_out == 1:
            yy = batch[2]
            a = sum(yy)
            b = len(yy)
            if a > 0:
                pos_weight = yy * b / a + np.ones_like(yy) - yy
            else:
                pos_weight = np.ones_like(yy)

        else:
            yy = batch[2]
            l = len(yy)
            a = np.sum(yy, axis=0)
            a = a / max(a)
            for ii in range(4):
                if a[ii] < 0.5:
                    a[ii] = 1
            pos_weight = np.sum(yy * a.reshape(-1, 1).T, axis=1)
    else:
        pos_weight = np.ones((2))
        pass
    train_weight(*batch, pos_weight.tolist())

    if curent_batch == batches_in_epoch:
        n = n + 1
        if weight:
            if n_out == 1:
                yy = Y_val
                a = sum(yy)
                b = len(yy)
                if a > 0:
                    pos_weight = yy * b / a + np.ones_like(yy) - yy
                else:
                    pos_weight = np.ones_like(yy)
            else:
                yy = Y_val
                l = len(yy)
                a = np.sum(yy, axis=0)
                a = a / max(a)
                for ii in range(4):
                    if a[ii] < 0.5:
                        a[ii] = 1
                pos_weight = np.sum(yy * a.reshape(-1, 1).T, axis=1)
        else:
            pos_weight = np.ones((2))
            pass
        loss_va, acc_va = evaluate_weight(x=X_val, fltr=Adj_val, y=Y_val, pos_weight=pos_weight.tolist())

        if loss_va < best_val_loss:
            best_val_loss = loss_va
            current_patience = patience
            loss_te, acc_te, _ = test_weight(x=X_test, fltr=Adj_test, y=Y_test)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break

        # Print results
        print(
            'Epochs: {:.0f} | ' 'Train loss: {:.4f}, acc: {:.4f} | ' 'Valid loss: {:.4f}, acc: {:.4f} | ' 'Test loss: {:.4f}, acc: {:.4f}'
            .format(n, train_loss_fn.result(), train_acc_fn.result(), loss_va, acc_va, loss_te, acc_te)
        )
        loss_train.append(train_loss_fn.result().numpy())
        acc_train.append(train_acc_fn.result().numpy())
        loss_val.append(loss_va)
        acc_val.append(acc_va)
        loss_test.append(loss_te)
        acc_test.append(acc_te)
        # Reset epoch
        train_loss_fn.reset_states()
        train_acc_fn.reset_states()
        curent_batch = 0

end = time.perf_counter()
print('training duration:%ss' % (end-start))
del X_train, X_val, Y_train, Y_val

EPOCHS = np.array(loss_train).shape[0]
HISTORY = np.zeros((6, EPOCHS))
HISTORY[0, :] = np.array(acc_train)
HISTORY[1, :] = np.array(acc_val)
HISTORY[2, :] = np.array(loss_train)
HISTORY[3, :] = np.array(loss_val)
HISTORY[4, :] = np.array(acc_test)
HISTORY[5, :] = np.array(loss_test)

#####################  Testing  ####################

print('\nTesting ------------')
weight = False

if label != 'both':
    loss, accuracy, Y_predict = test_weight(x=X_test, fltr=Adj_test, y=Y_test)
    acc_tpr = acc_fn_1(Y_test, Y_predict)
    acc_fn_1.reset_states()

    print('model test loss: ', loss)
    print('model test accuracy: ', accuracy)
    print('model test TPR: ', acc_tpr)

    Y_predict_int = np.rint(Y_predict)  # output
    con_mat = confusion_matrix(Y_test, Y_predict_int)  # confusion matrix
    print(con_mat)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    fpr, tpr, thresholds_keras = roc_curve(Y_test.astype(int), Y_predict)  # AUC
    auc = auc(fpr, tpr)
    print("AUC : ", auc)

    """
    save
    """
    path = './result_fre'
    if not os.path.exists(path + '/result_fre/%s/' % (exp_num)):
        os.makedirs(path + '/result_fre/%s/' % (exp_num))
    model.save_weights(path + '/result_fre/%s/model.h5' % (exp_num))

    f = h5py.File(path + '/result_fre/%s/histroy_%s_%s.h5' % (exp_num, chosedlength, label), 'w')
    f.create_dataset('train_history', data=HISTORY)
    f.create_dataset('test_loss', data=loss)
    f.create_dataset('test_accuracy', data=accuracy)
    f.create_dataset('test_matrix', data=con_mat)
    f.create_dataset('test_fpr', data=fpr)
    f.create_dataset('test_tpr', data=tpr)
    f.create_dataset('test_AUC', data=auc)
    f.create_dataset('pre', data=Y_predict)
    f.close()

else:
    loss, accuracy, Y_predict = test_weight(x=X_test, fltr=Adj_test, y=Y_test)

    print('model test loss: ', loss)
    print('model test accuracy: ', accuracy)

    Y_predict_int = np.rint(Y_predict)  # output
    con_mat = confusion_matrix(Y_test.argmax(axis=1), Y_predict_int.argmax(axis=1))  # confusion matrix
    print(con_mat)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    con_mat_norm = np.around(con_mat_norm, decimals=n_out)

    """
    save
    """
    path = './result_fre'
    if not os.path.exists(path + '/result_fre/%s/' % (exp_num)):
        os.makedirs(path + '/result_fre/%s/' % (exp_num))
    model.save_weights(path + '/result_fre/%s/model.h5' % (exp_num))

    f = h5py.File(path + '/result_fre/%s/histroy_%s_%s_%s.h5' % (exp_num, chosedlength, label, adj_mode), 'w')
    f.create_dataset('train_history', data=HISTORY)
    f.create_dataset('test_loss', data=loss)
    f.create_dataset('test_accuracy', data=accuracy)
    f.create_dataset('test_matrix', data=con_mat)
    f.create_dataset('pre', data=Y_predict)
    f.create_dataset('origin', data=Y_test)
    f.close()
