import numpy as np; np.random.seed(0)
import tensorflow as tf; tf.set_random_seed(seed=0)
import pandas as pd
import lightgbm as lgb
import gc
from collections import defaultdict
import matplotlib
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing import text, sequence
import numpy as np
from keras.constraints import max_norm
from keras.callbacks import *
from keras.initializers import Orthogonal
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import pickle
import gc; gc.enable()
import matplotlib 
import matplotlib.pyplot as plt
%matplotlib inline
import string
from scipy.stats import boxcox
import re
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.python.client import device_lib
from keras.layers import *
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, activations
from keras.metrics import *
from datetime import datetime
from keras_tqdm import TQDMNotebookCallback
from ipywidgets import IntProgress
from sklearn.model_selection import KFold, StratifiedKFold
import warnings; warnings.filterwarnings('ignore') 
from sklearn.metrics import accuracy_score, roc_auc_score

def transform_freq_feature(df1,df2,df3_base,feat):
    vc=df1[feat].append(df3_base[feat]).value_counts()
    df1[feat +"_freq"]= df1[feat].map(vc)
    df2[feat+"_freq"]= df2[feat].map(vc) 

def load_data(train, test, feature_cols):
    train_df = train[feature_cols].copy()
    test_df = test[feature_cols].copy()
    real_test_df = test[feature_cols].copy()

    unique_samples = []
    unique_count = np.zeros_like(test_df)
    for feature in range(test_df.shape[1]):
        _, index_, count_ = np.unique(test_df.values[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
    real_test_df=real_test_df.iloc[real_samples_indexes]
    
    for col in feature_cols:
        transform_freq_feature(train_df,test_df,real_test_df,col)
        
    for f in feature_cols: # normalzie
        vals = train_df[f].append(test_df.loc[te_real_samples_indexes,f]).values
        m, s = vals.mean(), vals.std()
        train_df[f] = (train_df[f]-m)/s
        test_df[f] = (test_df[f]-m)/s
    
    return train_df, test_df, real_samples_indexes

def build_model():
    # share components
    inputs = Input(shape=(200,2))
    
    main = inputs
    main = Dense(64, activation='relu')(main)
    main = Dense(32, activation='relu')(main)
    main = Flatten()(main)
    
    out = Dense(1, activation = 'sigmoid')(main) # 1 class to be classified

    model = Model(inputs, out)
    model.regularizers = [regularizers.l2(0.0001)]
    
    model.compile(optimizer = Adam(lr=0.001, clipnorm=1.), loss="binary_crossentropy")
    
    #model.summary()
    return model

class auc_score_monitor(Callback):
    def __init__(self, val_data, val_target, checkpoint_file, min_lr =1e-5, reduce_lr_patience=2, early_stop_patience=4, factor=0.1):
        self.val_data = val_data
        self.val_target = val_target
        self.checkpoint_file = checkpoint_file
        self.reduce_lr_patience = reduce_lr_patience
        self.early_stop_patience = early_stop_patience
        self.best_val_score = 0
        self.epoch_num = 0
        self.factor = factor
        self.unimproved_lr_counter = 0
        self.unimproved_stop_counter = 0
        self.min_lr = min_lr
        
    def on_train_begin(self, logs={}):
        self.val_scores = []
        
    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.val_data).reshape((-1,))
        val_score = roc_auc_score(self.val_target, val_pred)
        # clip pred
        self.val_scores.append(val_score)
        
        #print(self.val_target, '\n', val_pred)
        print('Epoch {} val_score: {:.5f}'.format(self.epoch_num, val_score))
        self.epoch_num += 1
        
        if val_score > self.best_val_score:
            print ('Val Score improve from {:5f} to {:5f}'.format(self.best_val_score, val_score))
            self.best_val_score = val_score
            self.unimproved_lr_counter = 0
            self.unimproved_stop_counter = 0
            if self.checkpoint_file is not None:
                print('Saving file to', self.checkpoint_file)
                self.model.save_weights(self.checkpoint_file)
        else:
            if val_score<self.best_val_score:
                print('no improve from {:.5f}'.format(self.best_val_score))
                self.unimproved_lr_counter += 1
                self.unimproved_stop_counter += 1
            
        if self.reduce_lr_patience is not None and self.unimproved_lr_counter >= self.reduce_lr_patience:
            current_lr = K.eval(self.model.optimizer.lr)
            if current_lr > self.min_lr:
                print('Reduce LR from {:.6f} to {:.6f}'.format(current_lr, current_lr*self.factor))
                K.set_value(self.model.optimizer.lr, current_lr*self.factor)
                #self.model.load_weights(self.checkpoint_file)
            else:
                pass
            
            self.unimproved_lr_counter = 0
            
        if self.early_stop_patience is not None and self.unimproved_stop_counter >= self.early_stop_patience:
            print('Early Stop Criteria Meet')
            self.model.stop_training = True
                
        return



def special_reshape(vals):
    return np.vstack([v.reshape((2,-1)).T.reshape((1, -1, 2)) for v in vals])

class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, positive_rate=1., negative_rate=1.,
                 pl_data=None, pl_soft_label=None, pl_sample_rate=1.):
        #'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.positive_rate = positive_rate
        self.negative_rate = negative_rate 
        self.pl_data = pl_data
        self.pl_soft_label = pl_soft_label
        self.pl_sample_rate = pl_sample_rate
        self.on_epoch_end()
        
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.resampled_y) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, len(self.resampled_y))
        indexes = np.arange(len(self.resampled_y))[start: end]

        # Generate data
        return self.resampled_X[indexes,:,:], self.resampled_y[indexes]

    def on_epoch_end(self):
        # resample + shuffle
        feat_len = 200
        
        if self.pl_data is not None:
            pl_idx = np.random.choice(np.arange(self.pl_data.shape[0]), 
                                      size=int(self.pl_data.shape[0]*self.pl_sample_rate), 
                                      replace=False)
            
            pl_y = self.pl_soft_label[pl_idx].copy()
            pl_x = self.pl_data[pl_idx,:].copy()
            
            pl_y_rank = pd.Series(pl_y).rank(ascending=False)
            filt = pl_y_rank<=int(len(pl_y)*.1) # mark top 10 % rank data as 1
            pl_y[filt] = 1.
            pl_y[~filt] = 0.
            
            X_p = np.concatenate([self.X[self.y==1], pl_x[pl_y==1]], axis=0)
            X_n = np.concatenate([self.X[self.y==0], pl_x[pl_y==0]], axis=0)
        else:    
            X_p = self.X[self.y==1]
            X_n = self.X[self.y==0]
        
        pos_size = int(self.positive_rate*X_p.shape[0])
        X_p_new = np.zeros((pos_size, X_p.shape[1])).astype(np.float32)
        neg_size = int(self.negative_rate*X_n.shape[0])
        X_n_new = np.zeros((neg_size, X_n.shape[1])).astype(np.float32)
        
        for f in range(feat_len):
            pos_idx = np.random.choice(np.arange(X_p.shape[0]), size=pos_size, replace=True)
            X_p_new[:, f] = X_p[pos_idx,f]
            X_p_new[:, f+feat_len] = X_p[pos_idx,f+feat_len]
            
            neg_idx = np.random.choice(np.arange(X_n.shape[0]), size=neg_size, replace=True)
            X_n_new[:, f] = X_n[neg_idx,f]
            X_n_new[:, f+feat_len] = X_n[neg_idx,f+feat_len]
            
        self.resampled_X = np.vstack([X_p_new, X_n_new])
        self.resampled_y = np.array([1]*pos_size+[0]*neg_size)
        
        seq = np.random.choice(np.arange(len(self.resampled_y)), size=len(self.resampled_y), replace=False)
        self.resampled_X = special_reshape(self.resampled_X[seq, :])
        self.resampled_y = self.resampled_y[seq]
        #print(self.resampled_X.shape, self.resampled_y.shape)

if __name__ == '__main__':

    train = pd.read_csv('data/train.csv.zip') # download this from kaggle websites
    test = pd.read_csv('data/test.csv.zip')

    special_cols = [col for col in train.columns if train[col].dtype != np.float64]
    feature_cols = [col for col in train.columns if col not in special_cols]
    target = train.target.values
    
    train_df, test_df, te_real_samples_indexes = load_data(train, test, feature_cols)
    
    # check gpu
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(device_lib.list_local_devices())
    print(K.tensorflow_backend._get_available_gpus())
    
    # configs for NN
    seed = 0
    train_epochs = 50
    batch_size=32 # 32 or 64 is good (too huge for my PC), 128 is worse in the past experiments
    cpu_count=4  
    n_classses = 1
    fold_num = 4
    model_prefix = 'nn-aug-v5' #'rnn-with-marcus-features-v4'
    bags = 10
    pseudo_label = False
    pseudo_label_sample_rate = 0.8
    
    # training models with several bags
    for b in range(bags):
        fold = 0  
        
        for tr_ix, val_ix in KFold(fold_num, shuffle=True, random_state=seed).split(target, target):    
            fold += 1

            print("fold = {}, bag = {}".format(fold, b))
            
            tr = train_df.values[tr_ix,:]
            tr_y = target[tr_ix]
            
            if pseudo_label:
                pseudo_train = test_df.values[real_samples_indexes,]
                pseudo_y = pd.read_pickle("oof+submission/nn-aug-v3_fold_4_seed_0_oof_test")[real_samples_indexes]
            else:
                pseudo_train = None
                pseudo_y = None
            
            val = special_reshape(train_df.values[val_ix,:])
            val_y = target[val_ix]

            model = build_model()
            file_path = "model_weights/{}_fold_{}_bag_{}.hdf5".format(model_prefix, fold, b)

            lrs = [0.001]*7+[0.0001]*10+[0.00001]*5
            lr_schd = LearningRateScheduler(lambda ep: lrs[ep], verbose=1)
            wmlog_loss_monitor = auc_score_monitor(val, val_y, 
                                                   checkpoint_file=None, reduce_lr_patience=None, early_stop_patience=None, 
                                                   factor=None) # calculate weighted m log loss per epoch

            training_generator = DataGenerator(tr, tr_y, batch_size=batch_size, positive_rate=2., negative_rate=1.,
                                               pl_data=pseudo_train, pl_soft_label=pseudo_y, pl_sample_rate=pseudo_label_sample_rate)
            history = model.fit_generator(generator=training_generator,
                                          validation_data=(val, val_y),
                                          use_multiprocessing=False,
                                          workers=1, 
                                          epochs=len(lrs),
                                          verbose = 0, 
                                          callbacks = [lr_schd,
                                                       wmlog_loss_monitor, TQDMNotebookCallback(leave_inner=True, leave_outer=True)])
            model.save_weights(file_path)
            del training_generator; gc.collect()
            K.clear_session()

    # generate oof + submission
    train_oof = np.zeros((train.shape[0],))
    test_oof = np.zeros((test.shape[0],))

    train_aucs = []

    model = build_model()
        
    for b in range(bags):
        fold=0
        for tr_ix, val_ix in KFold(fold_num, shuffle=True, random_state=seed).split(target, target):    
            fold += 1
            val = special_reshape(train_df.values[val_ix,:])
            val_y = target[val_ix]

            file_path = "model_weights/{}_fold_{}_bag_{}.hdf5".format(model_prefix, fold, b)

            # Predict val + test oofs
            model.load_weights(file_path) # load weight with best validation score

            pred = model.predict(val, batch_size=batch_size).reshape((len(val_ix),))
            train_oof[val_ix] += pred
            val_auc = roc_auc_score(target[val_ix], pred)
            train_aucs.append(val_auc)
            print('val acc = {:.5f}'.format(val_auc))

            test_oof += model.predict(special_reshape(test_df.values), batch_size=batch_size).reshape((test.shape[0],))/fold_num

    train_oof /= bags
    test_oof /= bags
    K.clear_session()
    
    full_auc = roc_auc_score(target, train_oof)
    print('CV Mean = {:.5f}, Std = {:.5f}, Overall AUC = {:.5f}'.format(np.mean(train_aucs), np.std(train_aucs), full_auc))
    
    pd.to_pickle(train_oof, "oof+submission/{}_fold_{}_seed_{}_oof_train".format(model_prefix, fold_num, seed))
    pd.to_pickle(test_oof, "oof+submission/{}_fold_{}_seed_{}_oof_test".format(model_prefix, fold_num, seed))
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub["target"] = test_oof
    sub.to_csv('oof+submission/' + model_prefix + '_' + str(full_auc).replace('.', '_') + ".csv", index=False)