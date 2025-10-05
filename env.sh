cd /mnt/42_store/xj/LongLive

# 1. 初始化conda
source /mnt/42_store/xj/anaconda3/etc/profile.d/conda.sh

# 2. 激活longlive环境
conda activate  longlive

export PYTHONNOUSERSITE=1
export PYTHONPATH=./:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH$
# export TORCH_CUDA_ARCH_LIST="8.0" # CUDA11.X，对应的算力为8.0

# export CUDA_VISIBLE_DEVICES=1