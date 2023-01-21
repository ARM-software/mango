import subprocess
import os
import shutil
import serial
import re
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K

#add your target hardware properties here
device_list = ["NUCLEO_F746ZG","NUCLEO_L476RG", "NUCLEO_F446RE", "ARCH_MAX", "Esense"]
arenaSize_list = [np.array([10,30,50,75,100,150,175,200,250,280,280]),
                  np.array([10,25,40,70,85,100,100]),
                  np.array([10,25,40,70,85,100,100]),
                  np.array([10,25,40,70,95,120,140,160,170,170]),
                  np.array([10,20,30,40,40])]
maxRAM_list = [300000, 100000,100000,180000,45000]
maxFlash_list = [800000,800000,400000,400000,15000000]

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def HIL_spec(dirpath="auritus_tcn/",chosen_device="NUCLEO_F746ZG",
             arenaSizes=np.arange(10,300,30),idx=0,window_size=200,number_of_channels=10,quantization=False):
    ser = serial.Serial('/dev/ttyACM0') #COM Port where board is attached
    RAM = -1
    Flash = -1
    Latency = -1
    err_flag = 0
    with cd(dirpath):
        with open('main.cpp') as f:
            z = f.readlines()
        f.close()
        z[25] = 'constexpr int kTensorArenaSize='+str(arenaSizes[idx])+'*1000;\n';
        z[13] = 'const int numSamples = '+str(window_size)+';\n';
        z[15] = 'const int numChannels = '+str(number_of_channels)+';\n';
        if(quantization==True):
            z[63] = 'input->data.f[samplesRead * numChannels + i] = rand()\n';
            z.insert(63,'srand(time(NULL));');
        else:
            z[63] = 'input->data.f[samplesRead * numChannels + i] = ((float)rand()/(float)(RAND_MAX))*5.0;\n'
        my_f = open("main.cpp","w")
        for item in z:
            my_f.write(item)
        my_f.close()
        if(os.path.exists(dirpath+"BUILD/") and os.path.isdir(dirpath+"BUILD/")):
            shutil.rmtree(dirpath+"BUILD/")
        os.system("mbed config root .")
        os.system("mbed deploy")
        cmd = "mbed compile -m "+ chosen_device +" -t GCC_ARM --flash"
        sp = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sp.communicate()
        x = out.decode("utf-8") 
        if(x.find("overflowed")==-1):
            RAM = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('Static RAM memory'):x.find('Total Flash memory')])][0]
            Flash = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('Total Flash memory'):x.find('Total Flash memory')+x[x.find('Total Flash memory'):].find('bytes')])][0]                 
            with serial.Serial('/dev/ttyACM0', 9600, timeout=20) as ser:
                s = ser.read(1000)
            x = s.decode("utf-8") 
            if(x.find("size is too small for all buffers")==-1 and x.find("to allocate")==-1 and x.find("missing")==-1 and x.find("Fault")==-1):
                Latency = [float(s) for s in re.findall(r'-?\d+\.?\d*', x[x.find('timer output'):x.find('ptimer output')+x[x.find('timer output'):].find('\n')])][0]   
                err_flag = 0; #no problem
            else:
                err_flag = 2; #arena too small or arena overflow - increase arena size
        elif x.find("counter backwards")!=-1:
            err_flag = 3; #RAM overflow during compilation - increase arena size
        else:
            err_flag = 1; #flash overflow error during compilation - choose smaller model
        print("HIL_SPEC_ERROR_FLAG: ",err_flag)
            
    return RAM, Flash, Latency, err_flag

def HIL_controller(dirpath="auritus_tcn/",
                   chosen_device="NUCLEO_F746ZG",window_size=200,number_of_channels=10,quantization=False):
    
    arenaSizes = arenaSize_list[device_list.index(chosen_device)]
    finRAM = -1
    finFlash = -1
    finLatency = -1
    idealArenaSize = -1
    masterError = 0
    j = 0
    while(masterError==0):
        RAM, Flash, Latency, err_flag = HIL_spec(dirpath,chosen_device,arenaSizes,j,
                                                 window_size,number_of_channels,quantization=False)
        print('Before change:',masterError)            
        if(err_flag == 0): #already got minimum required arena size, everything ok
            finRAM = RAM
            finFlash = Flash
            finLatency = Latency
            idealArenaSize = arenaSizes[j]
            masterError=1 
        elif(err_flag == 2 or err_flag ==3): #arena size too small, keep trying bigger arena
            print("Arena size: ",arenaSizes[j])
            finRAM = RAM
            finFlash = Flash
            finLatency = Latency
            idealArenaSize = arenaSizes[j]
            j+=1
        elif(j==len(arenaSizes)-1 and (err_flag == 2 or err_flag ==3)):  # arena size too small for maximum possible arena size 
            finRAM = -1
            finFlash = -1
            finLatency = -1
            idealArenaSize = -1
            masterError = 2
        else:  #flash overflow (err_flag == 1)
            finRAM = -1
            finFlash = -1
            finLatency = -1
            idealArenaSize = -1
            masterError = 3
        print("Arena size len,", len(arenaSizes))
        print('Err Flag:',err_flag)
        print('After change:',masterError)
            
    return finRAM, finFlash, finLatency, idealArenaSize, masterError
        
def get_model_memory_usage(batch_size, model): #SRAM Proxy
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    bytes_size = (total_memory + internal_model_mem_count)
    return bytes_size

def convert_to_tflite_model(model,training_data,quantization=False,output_name='g_model.tflite'):
    def representative_dataset():
        for i in range(100):
            yield(X[i].reshape(1,training_data.shape[1],training_data.shape[2]))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    if(quantization==True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset        
    model_tflite = converter.convert()
    open(output_name, "wb").write(model_tflite)

def convert_to_cpp_model(dirpath="auritus_tcn/"):
    os.system("xxd -i g_model.tflite > model.cc")
    with open('model.cc') as f:
        z = f.readlines()
    f.close()   
    z.insert(0,'#include "model.h"\n')
    z = [w.replace('unsigned int','const int') for w in z]
    z = [w.replace('g_model_tflite','g_model') for w in z]
    z = [w.replace('unsigned char','alignas(8) const unsigned char') for w in z]
    my_f = open("model.cc","w")
    for item in z:
        my_f.write(item)
    my_f.close()
    h_file_cont = ['#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n',
               '#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n',
              'extern const unsigned char g_model[];\n',
              'extern const int g_model_len;\n',
              '#endif\n']
    my_f = open("model.h","w")
    for item in h_file_cont:
        my_f.write(item)
    my_f.close()
    shutil.copy('model.cc',dirpath+'model.cc')
    shutil.copy('model.h',dirpath+'model.h')
    
def return_hardware_specs(hardware):
    maxRAM = maxRAM_list[device_list.index(hardware)]
    maxFlash = maxFlash_list[device_list.index(hardware)]
    return maxRAM, maxFlash
    
    
    
    
