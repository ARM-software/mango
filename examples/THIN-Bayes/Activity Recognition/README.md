# Activity Recognition with Platform-Aware NAS

This repository explores developing human activity detection classifiers for low-end IoT devices using platform-aware NAS called THIN-Bayes, designed through ARM Mango. We use a temporal convolutional network (TCN) as the backbone architecture.

## Code Architecture

- Download the activity dataset from: https://github.com/swapnilsayansaha/tinyml_security/tree/main/Model%20Training/Human%20Activity%20Recognition/Data
- [```auritus_tcn```](https://drive.google.com/file/d/1htTsLH4gPPdamwP6WmqyMqU4YzgbspJd/view?usp=sharing) has actual Tensorflow Lite Micro style C++ code that can be run on Mbed-enabled boards. Please place it in your home directory in the Mbed programs folder (e.g., ```home/nesl/Mbed Programs/auritus_tcn```). Refer to the TFLM guide to understand how ```main.cpp``` works: https://www.tensorflow.org/lite/microcontrollers
- You can add your own Mbed-enabled target hardware in ```hardware_utls.py```. You need to know the maximum amount of RAM, maximum amount of Flash and a list of arena sizes you want to optimize for for the hardware.
- The Jupyter notebook is well-commented to guide you through the NAS process. One particular thing to note is how the score weighs each optimization variable (accuracy, RAM, flash, latency). You can play around with the weights.
- The scripts are written to be trained on GPU. If you do not have GPU, first comment this line in each notebook: ```os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"```, then change the next line as follows: ```os.environ["CUDA_VISIBLE_DEVICES"]="-1"```

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

Saha S.S., Sandha S.S., Garcia L., Srivastava M. (2022) "TinyOdom: Hardware-Aware Efficient Neural Inertial Navigation". (Under Review) Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (2022).
