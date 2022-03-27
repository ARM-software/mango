import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, atan2, sin, cos, radians
import h5py
import json
import glob
from scipy.signal import savgol_filter
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
import os
from pydometer import Pedometer
import quaternion
import math
import geometry_helpers
import math

def import_oxiod_dataset(type_flag = 2, useMagnetometer = True, useStepCounter = True, AugmentationCopies = 0,
                         dataset_folder = 'oxiod/',
                         sub_folders = ['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                         sampling_rate = 100, window_size = 200, stride = 10, verbose=False):
 
    default_channels = ['Timestamp','Roll','Pitch','Yaw','Gyro_X','Gyro_Y','Gyro_Z','Grav_X','Grav_Y','Grav_Z','Lin_Acc_X','Lin_Acc_Y','Lin_Acc_Z','Mag_X','Mag_Y','Mag_Z']
    default_GT_channels = ['Timestamp','Header','Pose_X','Pose_Y','Pose_Z','Rot_X','Rot_Y','Rot_Z','Rot_W']
    wanted_GT_channels = ['Pose_X','Pose_Y','Pose_Z']
    if(type_flag == 1): #full training set (including validation)
        type_file = 'Train_Valid.txt'
    elif(type_flag==2): #training set
        type_file = 'Train.txt'
    elif(type_flag==3): #validation set
        type_file = 'Valid.txt'
    elif(type_flag==4): #test set
        type_file = 'Test.txt'
    
    if(useMagnetometer):
        channel_count = 9
    else:
        channel_count = 6
    
    X_orig = np.empty([0,channel_count])
    x0_list = []
    y0_list = []
    size_of_each = []
    X = np.empty([0, window_size, channel_count])
    Y_disp = np.empty([0])
    Y_head = np.empty([0])
    Y_pos = np.empty([0,window_size, 2])
    x_vel = np.empty([0])
    y_vel = np.empty([0])
    head_s = np.empty([0])
    head_c = np.empty([0])
    
    if(useStepCounter):
        loc_3D_mat = np.empty([0,window_size])
       
    for folder in tqdm(sub_folders):
        with open(dataset_folder+folder+type_file, 'r') as f:
            list_of_files = [line.strip() for line in f]
        for line in list_of_files:
            if(verbose==True):
                print('Processing for (file and ground truth): '+folder+line)
            cur_train = pd.read_csv(dataset_folder+folder+line,header=None)
            cur_train.columns = default_channels
            acc_x = cur_train['Lin_Acc_X'].to_numpy() + cur_train['Grav_X'].to_numpy()
            acc_y = cur_train['Lin_Acc_Y'].to_numpy() + cur_train['Grav_Y'].to_numpy()
            acc_z = cur_train['Lin_Acc_Z'].to_numpy() + cur_train['Grav_Z'].to_numpy()
            gyr_x = cur_train['Gyro_X'].to_numpy().reshape((acc_x.shape[0],1))
            gyr_y = cur_train['Gyro_Y'].to_numpy().reshape((acc_x.shape[0],1))
            gyr_z = cur_train['Gyro_Z'].to_numpy().reshape((acc_x.shape[0],1))
            
            if(useMagnetometer):
                mag_x = cur_train['Mag_X'].to_numpy().reshape((acc_x.shape[0],1))
                mag_y = cur_train['Mag_Y'].to_numpy().reshape((acc_x.shape[0],1))
                mag_z = cur_train['Mag_Z'].to_numpy().reshape((acc_x.shape[0],1)) 
                
            if(useStepCounter):
                p = Pedometer(gx=acc_x, gy=acc_y, gz=acc_z, sr=sampling_rate)
                step_count, step_locations = p.get_steps()
                loc = np.zeros(cur_train.shape[0])
                loc[step_locations] = 1
           
            acc_x = acc_x.reshape((acc_x.shape[0],1))
            acc_y = acc_x.reshape((acc_x.shape[0],1))
            acc_z = acc_x.reshape((acc_x.shape[0],1))
           
            if(useMagnetometer):
                cur_train = np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,mag_x,mag_y,mag_z),axis=1)
            else:
                cur_train = np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z),axis=1)
                                           
            cur_GT = pd.read_csv(dataset_folder+folder+line.replace('imu','vi'),header=None)
            cur_GT.columns = default_GT_channels
            cur_GT.drop(list(set(default_GT_channels) - set(wanted_GT_channels)), axis=1,inplace=True)
            cur_GT = cur_GT.to_numpy()
   
            windows = SlidingWindow(size=window_size, stride=stride)
            cur_train_3D = windows.fit_transform(cur_train[:,0])
            for i in range(1,cur_train.shape[1]):
                X_windows = windows.fit_transform(cur_train[:,i])
                cur_train_3D = np.dstack((cur_train_3D,X_windows))
            
            if(useStepCounter):   
                loc_3D = windows.fit_transform(loc)

            cur_GT_3D = windows.fit_transform(cur_GT[:,0])
            for i in range(1,cur_GT.shape[1]):
                X_windows = windows.fit_transform(cur_GT[:,i])
                cur_GT_3D = np.dstack((cur_GT_3D,X_windows))  
           
            vx = np.zeros((cur_GT_3D.shape[0]))
            vy = np.zeros((cur_GT_3D.shape[0]))
            
            heading_s = np.zeros((cur_GT_3D.shape[0]))
            heading_c = np.zeros((cur_GT_3D.shape[0]))
            for i in range(cur_GT_3D.shape[0]):    
                s,c = abs_heading_sin_cos(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
                heading_s[i] = s
                heading_c[i] = c            
           
            displacement_GT_abs = np.zeros(cur_GT_3D.shape[0])
            heading_GT = np.zeros((cur_GT_3D.shape[0]))
            prev = 0
            for i in range(cur_GT_3D.shape[0]):  
                Xdisp = (cur_GT_3D[i,-1,0]-cur_GT_3D[i,0,0])
                vx[i] = Xdisp
                Ydisp = (cur_GT_3D[i,-1,1]-cur_GT_3D[i,0,1])
                vy[i] = Ydisp
                displacement_GT_abs[i] = sqrt((Xdisp**2) + (Ydisp**2))  
                theta = abs_heading(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
                if theta<180:
                    theta = theta + 180
       
                heading_GT[i] = theta - prev
                if(heading_GT[i]>100 or heading_GT[i]<-100):
                    theta2 = theta
                    prev2 = prev
                    if theta<prev:
                        theta2 = theta + 360
                    else:
                        prev2 =  prev + 360
                    heading_GT[i] = theta2 - prev2
                prev = theta
            
            X = np.vstack((X, cur_train_3D))
            X_orig = np.concatenate((X_orig,cur_train))
            Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
            Y_head = np.concatenate((Y_head, heading_GT))
            Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
            x0_list.append(cur_GT[0,0])
            y0_list.append(cur_GT[0,1])
            size_of_each.append(cur_GT_3D.shape[0])
            x_vel = np.concatenate((x_vel, vx))
            y_vel = np.concatenate((y_vel, vy))
            head_s = np.concatenate((head_s,heading_s))
            head_c = np.concatenate((head_c,heading_c))
            if(useStepCounter):
                loc_3D_mat = np.vstack((loc_3D_mat,loc_3D))
            if(AugmentationCopies>0):
                for i in range(AugmentationCopies):
                    out = random_rotate(cur_train_3D, useMagnetometer)
                    X = np.vstack((X, out))
                    X_orig = np.concatenate((X_orig,cur_train))
                    Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
                    Y_head = np.concatenate((Y_head, heading_GT))
                    Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
                    x0_list.append(cur_GT[0,0])
                    y0_list.append(cur_GT[0,1])
                    size_of_each.append(cur_GT_3D.shape[0])
                    x_vel = np.concatenate((x_vel, vx))
                    y_vel = np.concatenate((y_vel, vy))
                    head_s = np.concatenate((head_s,heading_s))
                    head_c = np.concatenate((head_c,heading_c))
                    if(useStepCounter):
                        loc_3D_mat = np.vstack((loc_3D_mat,loc_3D))               
           
    if(useStepCounter):
        X = np.concatenate((X,loc_3D_mat.reshape(loc_3D_mat.shape[0],loc_3D_mat.shape[1],1)),axis=2)
    
    #returns 1. training set from IMU 2. ground truth displacements 3. ground truth heading rates 4. ground truth position
    # 5. list of initial x positions 6. list of initial y positions 7. size of each file in windowed form
    # 8. ground truth x velocity 9. ground truth y velocity 10. heading rate in terms of sin 11. heading rate in terms of cos
    # 12. unwindowed training set from IMU
    return  X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, head_s, head_c, X_orig

def abs_heading(cur_x, cur_y, prev_x, prev_y):
        dely = (cur_y - prev_y)
        delx = (cur_x - prev_x)
        delh= atan2(delx,dely)*57.2958
        return delh
    
def abs_heading_sin_cos(cur_x, cur_y, prev_x, prev_y):
        dely = (cur_y - prev_y)
        delx = (cur_x - prev_x)
        sqr = np.sqrt(dely*dely + delx*delx)
        s = dely/sqr
        c = delx/sqr
        return s,c

def random_rotate(input,useMagnetometer=True):
    output = np.copy(input)
    euler = np.random.uniform(0, np.pi, size=3)
    for i in range(0, input.shape[0]):
        input_acc = input[i,:,0:3]
        input_rot = input[i,:,3:6]
        if(useMagnetometer):
            input_mag = input[i,:,6:9]  
        Rot = geometry_helpers.euler2mat(euler[0],euler[1],euler[2])
        output_acc = np.dot(Rot, input_acc.T).T
        output_rot = np.dot(Rot, input_rot.T).T
        if(useMagnetometer):
            output_mag = np.dot(Rot, input_mag.T).T
            output[i,:,:] = np.hstack((output_acc, output_rot, output_mag))  
        else:
            output[i,:,:] = np.hstack((output_acc, output_rot))    
    return output

def orientation_to_angles(ori):
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles


def adjust_angle_array(angles):
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def Cal_TE(Gvx, Gvy, Pvx, Pvy, sampling_rate=100,window_size=200,stride=10,length=None):
    
    if length==None:
        length = len(Gvx)
        
    distance = []
    
    for i in range(length):
        d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i]))
        d = math.sqrt(d)
        distance.append(d)
    
    mean_distance = sum(distance)/len(distance)
    ate = mean_distance
    at_all = distance
    
    n_windows_one_min= int(((sampling_rate*60)-window_size)/stride)
    distance = []
    if(n_windows_one_min < length):
        for i in range(n_windows_one_min):
            d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i]))
            d = math.sqrt(d)
            distance.append(d)
        rte = sum(distance)/len(distance)
    else:
        rte=ate*(n_windows_one_min/length)
    
    rt_all = distance
    return ate, rte, at_all, rt_all

def Cal_len_meters(Gvx, Gvy, length=None):
    if length==None:
        length = len(Gvx)
        
    distance = []
    
    for i in range(1, length):
        d = ((Gvx[i]-Gvx[i-1])*(Gvx[i]-Gvx[i-1])) + ((Gvy[i]-Gvy[i-1])*(Gvy[i]-Gvy[i-1]))
        d = math.sqrt(d)
        distance.append(d)
    
    sum_distance = sum(distance)
    
    return sum_distance    
    
