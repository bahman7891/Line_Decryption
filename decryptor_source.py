#!/usr/bin/python


'''
Deep learning the encrypted sentences from texts by fragmentation.
Author: Bahman Roostaei.
Rights: All rights are reserved to the author. Please do not share.
Creation date: 30.June.2017
Contact: bmn7891@gmail.com
'''


import sys
import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input



def pad_encode(string,max_len):
    '''
    adds '0' to make string length equal to max_len or chops it if the string has
    length larger than max_len and adds '0' to the remaining chop.

    '''    
    string_len = len(string)
    if string_len < max_len:
        diff = max_len - string_len
        string += '0'*diff
        return [encoder(string)]
    elif string_len > max_len:
            string_raw = string
            remain_len = len(string_raw)
            string_chops = []
            while remain_len > 0:
                string_chops.append(string_raw[:max_len])
                string_raw = string_raw[max_len:]
                remain_len = len(string_raw)

            padded_encoded = [pad_encode(s,max_len) for s in string_chops]
            return padded_encoded
    else:
        return [encoder(string)]

def encoder(string):
    """
    return a one-hot encoded array for 26 letters and '0' as 27th.
    """
    ref = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
            's','t','u','v','w','x','y','z','0']
        
    encoded = np.zeros((416,27))
    for j,lett in enumerate(string):
        if lett != '0':
            encoded[j,ref.index(lett)] = 1
    return encoded



def create_network(max_len, **model_params):
    '''
    create the architecture of the model.
    
    '''
    
    inputs = Input(shape=(416,27))
    
    convo_1 = Convolution1D(model_params['conv_1']['num_filters'],
                            border_mode='valid',
                            filter_length=model_params['conv_1']['filter_length'],
                            activation="relu",
                            subsample_length=model_params['conv_1']['subsample_length'],
                            name=model_params['conv_1']['name'])(inputs)
    

    l_lstm = LSTM(model_params['lstm_1']['output_dim'],
                  return_sequences = True,
                  go_backwards= False,
                  name=model_params['lstm_1']['left_lstm']['name'])(convo_1)
    

    flat = Flatten(name=model_params['flatten_1']['name'])(l_lstm)

    dense_1 = Dense(model_params['dense_1']['output_dim'],
                    activation='relu',
                    name=model_params['dense_1']['name'])(flat)
    
    outputs = Dense(model_params['dense_2']['output_dim'],
                    activation='sigmoid',
                    name=model_params['dense_2']['name'])(dense_1)
    
    return (inputs, outputs)




if __name__ == "__main__":
    
    print('Reading train and test files ...')
    
    with open('xtrain_obfuscated.txt') as f:
        X_train = f.readlines()
    
    with open('ytrain.txt') as f:
        y_train = f.readlines()
    
    with open('xtest_obfuscated.txt') as f:
        X_test = f.readlines()
    
    print('done.')
    print('processing training data ...')
    
    # collecting the fragments pad them with '0' and encoding.
    X_train_collect = [pad_encode(x.strip(),416) for x in X_train]
    
    # adjusting the train and test data.
    X_train_encoded = []
    y_train_collect = []
    for i,x in enumerate(X_train_collect):
        if len(x) ==1:
            X_train_encoded.append(x[0])
            y_train_collect.append(y_train[i].strip())
        elif len(x) > 1:
            for sub_x in x:
                X_train_encoded.append(sub_x[0])
                y_train_collect.append(y_train[i].strip())
                
    y_train_encoded = np.zeros((len(y_train_collect),12))
    for j,y in enumerate(y_train_collect):
        y_train_encoded[j,int(y)] = 1
    
    X_train_np = np.asarray(X_train_encoded)
    y_train_np = np.asarray(y_train_encoded)
    
    # defining the parameters of the architecture.
    
    num_classes = 12
    model_params = {
        
        'conv_1': 
        { 
            'num_filters': 10,
            'filter_length': 5,
            'subsample_length': 4,
            'name': 'multiclass_1dconv_1',
         },
    
        'lstm_1':
        {
            'output_dim': 10,
            'left_lstm': 
            {
                'name': 'multiclass_l_lstm_1'
            },
            'right_lstm':
            {
                'name': 'multiclass_r_lstm_1'
            },
        },
        
        'flatten_1':
        {
            'name': 'multiclass_flatten_1'
        },
        'dense_1':
        {
            'output_dim': 100,
            'name': 'multiclass_1'
        },
        
         'dense_2':
        {
            'output_dim': num_classes,
            'name': 'multiclass_dense_2'
        },

        'num_classes' :num_classes,
        'num_time_steps':27
    }
    
    print('Creating model ...')
    
    model = Model(*create_network(max_len=416, **model_params))

    train = input('Please choose: train the model[y] or use trained weights[n]:')
    
    if train == 'y':
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath="bestmodel_end2end.hdf5", verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        model.fit(X_train_np[:30000], y_train_np[:30000], batch_size=100,verbose=1,shuffle=True,nb_epoch=1)   
        
    else:
        model.load_weights('weights')
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Evaluating model.')
    loss_and_metrics = model.evaluate(X_train_np[30000:], 
                                  y_train_np[30000:], batch_size=100)
    print('loss:',loss_and_metrics[0])
    print('accuracy:',loss_and_metrics[1])
    
    
    print('processing test data ...')
    X_test_collect = [pad_encode(x.strip(),416) for x in X_test]
    X_test_encoded = []
    y_predict = []
    print('done.')
    print('Predicting the test results. This may take about 5 minutes ...')
    for x in X_test_collect:
        if len(x) ==1:
            x_shape = x[0].shape
            X_test_encoded = x[0].reshape(1,x_shape[0],x_shape[1])
            y_pred = model.predict(X_test_encoded)
            y_predict.append(np.argmax(y_pred))
        elif len(x) > 1:
            preds = []
            max_ind = 0
            max_pred = 0
            for sub_x in x:
                sub_x_shape = sub_x[0].shape
                pred = model.predict(sub_x[0].reshape(1,sub_x_shape[0],sub_x_shape[1]))   
                if max(pred[0]) > max_pred:
                    max_pred = max(pred[0])
                    max_ind = np.argmax(pred)
            y_predict.append(max_ind)
    print('done. Writing ...')
    with open('ytest.txt','w') as f:
        for y in y_predict:
            f.writelines(str(y)+'\n')
    print('done. Results are printed in the file ytest.txt .')



