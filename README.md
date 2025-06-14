# PQSC Environment Building
Step 1:

pip install tensorflow-gpu==2.10.0

conda install cudatoolkit==11.3.1 cudnn==8.2.1
(or)
conda install cudatoolkit==11.2
conda install -c conda-forge cudnn=8.1.0


If you meet error "tensorflow  can not find cudart.so", add the lib path to your running environment, e.g.,
LD_LIBRARY_PATH=.../your conda name/envs/your env name/lib

Step 2:

pip install sionna 

(It will automatically uninstall some of the existing packages according to Sionna's preferences and install its default versions; incompatible versions need to be adjusted manually.)

Step 3:

conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch

Following might help in debugging：
CUDA_ARCH="750" (for GTX2080); （ if you meet Error "drjit compiler failure: cuda_check(): API error 0200", it indicates that drjit finds a different GPU framework, and a corresponding CUDA_ARCH is needed）
LD_LIBRARY_PATH=.../your conda name/envs/your env name/lib; 
