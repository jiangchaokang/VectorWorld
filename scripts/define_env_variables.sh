export PYTHONPATH=$(pwd):$PYTHONPATH
export PROJECT_ROOT=$(pwd) 
export DATASET_ROOT=$SCRATCH_ROOT
export CONFIG_PATH=$PROJECT_ROOT/configs
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True