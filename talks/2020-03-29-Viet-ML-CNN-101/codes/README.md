# talks / 2020-03-29-Viet-ML-CNN-101 / codes

The demo only work on Python 3.5–3.7

Inside this folder, create a virtual env:
```
python -m venv ./venv
```
Activate venv
```
./venv/Scripts/activate
```
Install requirement packages
```
python -m pip install --upgrade pip
pip install -r ./requirements.txt
```
To run demos from command line
```
python ./<file_to_run.py>
```

If you have a Nvidia GPU, you can install the gpu version of tensorflow via 
```
pip install tensorflow-gpu
```
Make sure you have installed the cudnn and requirement drivers. 
Ref: https://www.tensorflow.org/install/gpu
```
Download and update driver of your Nvidia card.
Cuda:
    Download and install Cuda 10.0 toolkit (or whatever version depend on your card)
    In the installation directory (it should be: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0):
        Move cupti.lib from extras/CUPTI/libx64/ to lib/x64/
        Move cupti64_90.dll from extras/CUPTI/libx64/ to /bin/
CuDNN:
    Download CuDNN : https://developer.nvidia.com/rdp/cudnn-download
        Create an account is obligated.
        Make sure to install the latest 7.x.x release that is compatible with your CUDA version (e.g. 10.0).
        Merge the content of the downloaded zip file into the existing CUDA stuff in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0.
```
