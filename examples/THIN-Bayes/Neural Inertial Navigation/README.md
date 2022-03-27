#  Neural Inertial Navigation with Platform-Aware NAS

This repository explores developing neural inertial navigation models for low-end IoT devices using platform-aware NAS called THIN-Bayes, designed through ARM Mango. We use a temporal convolutional network (TCN) as the backbone architecture.

## Code Architecture

- Check the guide inside ```dataset_download_and_splits``` for download link to dataset and splits.
- The notebook is  divided into the following parts: 1. Data import (training, validation, test) 2. Training and NAS 3. Training the best model 4. Evaluation of best model on test set and sample plots 5. Deployment on real-hardware.
- ```tinyodom_tcn``` has actual Tensorflow Lite Micro style C++ code that can be run on Mbed-enabled boards. You must place it in your home directory in the Mbed programs folder (e.g., ```home/nesl/Mbed Programs/tinyodom_tcn```) if you want to run platform-in-the-loop NAS. Refer to the TFLM guide to understand how main.cpp works: https://www.tensorflow.org/lite/microcontrollers
- The script is written to be trained on GPU. If you do not have GPU, first comment this line in each notebook: ```os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"```, then change the next line as follows: ```os.environ["CUDA_VISIBLE_DEVICES"]="-1"```
- The Jupyter notebook is well-commented to guide you through the NAS process. One particular thing to note is how the score weighs each optimization variable (accuracy, RAM, flash, latency). You can play around with the weights.
- You can add your own Mbed-enabled target hardware in ```hardware_utls.py```. You need to know the maximum amount of RAM, maximum amount of Flash and a list of arena sizes you want to optimize for for the hardware.

## Required items 
- A GPU Workstation running Ubuntu 20.04.
- Python 3.8+ must be installed, preferably through Anaconda or Virtualenv, https://docs.conda.io/en/latest/, https://virtualenv.pypa.io/en/latest/
- Python package requirements are listed in ```requirements.txt```. Please install them before running the Python scripts. Note that Tensorflow 2.5.0 is a must for working with the TinyML model scripts. Tensorflow 1.x would not work.
- Couple of STM32 Nucleo Boards (must be Mbed enabled) for platform-aware NAS, https://www.st.com/en/evaluation-tools/stm32-nucleo-boards.html, https://os.mbed.com/platforms/
- Mbed Studio, https://os.mbed.com/studio/
- C/C++ for compiler Mbed CLI and conversion of TinyML models to C (your computer will generally come with one).
- GNU ARM Embedded Toolchain (for Mbed CLI), https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm

## Citation
Please cite this as:

Saha S.S., Sandha S.S., Pei S., Jain V., Wang Z., Li Y., Sarker A., Srivastava M. (2022) Auritus - An Open Source Optimization Toolkit for Training and Deployment of Human Movement Models and Filters Using Earables. (Under Review) Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2022).




