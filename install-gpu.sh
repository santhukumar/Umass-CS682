# This script is designed to work with ubuntu 16.04 LTS

# ensure system is updated and has basic build tools
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common


# download and install GPU drivers
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" -O "cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"

sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-8-0
# you should not see any errors with below command
sudo modprobe nvidia
nvidia-smi
# My output to nvidia-smi is 
# Wed Oct 18 04:02:36 2017
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 384.81                 Driver Version: 384.81                    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   40C    P0    69W / 149W |  10947MiB / 11439MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID   Type   Process name                             Usage      |
# |=============================================================================|
# |    0      2157      G   /usr/lib/xorg/Xorg                            15MiB |
# |    0      2779      C   /home/apple/anaconda3/bin/python           10918MiB |
# +-----------------------------------------------------------------------------+


#installing cudnn6
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}echo 'export PATH=$PATH:/usr/local/cuda/bin' >> $HOME/.profile; source $HOME/.profile
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH' >> ~/.profile; source ~/.profile
# You can check if cudnn is installed properly with below command
# $ find /usr | grep libcudnn
# /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.6.0.21
# /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn_static.a
# /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.6
# /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so


#tensor flow recommends us to use virtual env instead of using /usr/bin/python
sudo apt-get install virtualenv
sudo apt-get install python-setuptools
sudo easy_install virtualenv
virtualenv $HOME/venv
echo 'source venv/bin/activate' >> $HOME/.profile

pip install tensorflow-gpu
# If tensor flow is installed properly you can check the installation with below command. 
# $ python -c 'from tensorflow.python.client import device_lib; print device_lib.list_local_devices()'

# 2017-10-18 04:06:41.970832: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-10-18 04:06:41.970873: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-10-18 04:06:41.970880: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2017-10-18 04:06:41.970885: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-10-18 04:06:41.970890: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# 2017-10-18 04:06:42.073168: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2017-10-18 04:06:42.073755: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 0000:00:04.0
# Total memory: 11.17GiB
# Free memory: 432.31MiB
# 2017-10-18 04:06:42.073783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
# 2017-10-18 04:06:42.073793: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
# 2017-10-18 04:06:42.073805: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
# [name: "/cpu:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 5603554496112123154
# , name: "/gpu:0"
# device_type: "GPU"
# memory_limit: 217382912
# locality {
#   bus_id: 1
# }
# incarnation: 1888618465468897043
# physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0"
# ]


# install Anaconda and python3 for current user
mkdir downloads
cd downloads
wget "https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh" -O "Anaconda3-5.0.0.1-Linux-x86_64.sh"
bash "Anaconda3-5.0.0.1-Linux-x86_64.sh" -b


echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all
# Check Anakonda and python installation
# $ conda -V
# conda 4.3.29
# $ python -V
# Python 2.7.12
# $ python3 -V
# Python 3.6.2 :: Anaconda, Inc.


# install and configure theano
pip install theano
echo "[global]
device = gpu
floatX = float32

[cuda]
root = /usr/local/cuda" > ~/.theanorc
# Check theano installation
# $ pip show theano
# Name: Theano
# Version: 0.9.0
# Summary: Optimizing compiler for evaluating mathematical expressions on CPUs and GPUs.
# Home-page: http://deeplearning.net/software/theano/
# Author: LISA laboratory, University of Montreal
# Author-email: theano-dev@googlegroups.com
# License: BSD
# Location: /home/apple/venv/lib/python2.7/site-packages
# Requires: six, scipy, numpy


# install and configure keras
pip install keras
mkdir ~/.keras
echo '{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}' > ~/.keras/keras.json
# $ pip show keras
# Name: Keras
# Version: 2.0.8
# Summary: Deep Learning for Python
# Home-page: https://github.com/fchollet/keras
# Author: Francois Chollet
# Author-email: francois.chollet@gmail.com
# License: MIT
# Location: /home/apple/venv/lib/python2.7/site-packages
# Requires: pyyaml, six, scipy, numpy


# keras needs h5py
pip install h5py


# configure jupyter and prompt for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py




# References
# http://forums.fast.ai/t/py3-and-tensorflow-setup/1460/39
# http://forums.fast.ai/t/so-youre-ready-to-graduate-to-part-2/5978
# https://github.com/lwneal/install-keras
# https://gist.github.com/mjdietzx/0ff77af5ae60622ce6ed8c4d9b419f45
# https://stackoverflow.com/