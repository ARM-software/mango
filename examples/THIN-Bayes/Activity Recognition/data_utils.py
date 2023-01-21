import os
import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
from tqdm import tqdm
import random
from tensorflow.keras.utils import to_categorical

def import_auritus_activity_dataset(dataset_folder = 'Train_Val_test/', use_timestamp=False, shuffle=True, window_size = 550, stride = 50, return_test_set = True, test_set_size = 300,channels=0):
    if(use_timestamp==True and channels==0):
        X_tr = np.empty([0, window_size, 7])
    elif(use_timestamp==False and channels==2):
        X_tr = np.empty([0, window_size, 2])
    elif(use_timestamp==True and channels==2):
        X_tr = np.empty([0, window_size, 3])
    else:
        X_tr = np.empty([0, window_size, 6])
    Y_tr = np.empty([0,1])
    train_file_list = os.listdir(dataset_folder)
    if('.DS_Store' in train_file_list):
        train_file_list.remove('.DS_Store')
    labels = ['W','R','J','St','Tl','Tr','Si','L','F']
    for line in tqdm(train_file_list):
        if(use_timestamp==True):
            cur_train = pd.read_csv(dataset_folder+line,header=None,usecols=[0,1,2,3,4,5,6]).to_numpy()
        else:
            cur_train = pd.read_csv(dataset_folder+line,header=None,usecols=[0,1,2,3,4,5]).to_numpy()
        if(channels==2):
            acc = np.sqrt(cur_train[:,0]*cur_train[:,0] + cur_train[:,1]*cur_train[:,1] +cur_train[:,2]*cur_train[:,2])
            gyr = np.sqrt(cur_train[:,3]*cur_train[:,3] + cur_train[:,4]*cur_train[:,4] +cur_train[:,5]*cur_train[:,5])
            if(use_timestamp==False):
                cur_train = np.transpose(np.vstack((acc,gyr)))
            else:
                time_vec = cur_train[:,6]
                cur_train = np.transpose(np.vstack((acc,gyr,time_vec)))
        cur_label = labels.index([ele for ele in labels if(ele in line)][0])
        windows = SlidingWindow(size=window_size, stride=stride)
        cur_train_3D = windows.fit_transform(cur_train[:,0])
        for i in range(1,cur_train.shape[1]):
            X_windows = windows.fit_transform(cur_train[:,i])
            cur_train_3D = np.dstack((cur_train_3D,X_windows))  
        cur_GT = cur_label * np.ones((cur_train_3D.shape[0],1))
        X_tr = np.vstack((X_tr, cur_train_3D))
        Y_tr = np.concatenate((Y_tr, cur_GT))
    Y_tr = Y_tr.flatten()
    if(shuffle==True):
        shuffler = np.random.permutation(X_tr.shape[0])
        X_tr = X_tr[shuffler]
        Y_tr = Y_tr[shuffler]
        Y_tr = to_categorical(Y_tr,9)
    if(return_test_set==True):
        idx = np.random.randint(X_tr.shape[0]-test_set_size, size=1)[0]
        X_test = X_tr[idx:idx+test_set_size]
        Y_test = Y_tr[idx:idx+test_set_size]
        X_tr = np.delete(X_tr,np.arange(idx,idx+test_set_size),axis=0)
        Y_tr = np.delete(Y_tr,np.arange(idx,idx+test_set_size),axis=0)

    if(return_test_set==True):
        return X_tr, Y_tr, X_test, Y_test
    else:
        return X_tr, Y_tr



