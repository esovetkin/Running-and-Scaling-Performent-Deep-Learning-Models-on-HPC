#!/usr/bin/env bash

#SBATCH --account=training2560
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

set -xe

export curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
export curr_dir=$(dirname "${curr_file}")
export root_dir=$(realpath "${curr_dir}/..")

# We activate our environemnt
source /p/scratch/atmlaml/HPC-Supporters-Course/sc_venv_template_HPC_supporter_course/activate.sh
# The above path is a virtual environment that was created and tested previously.
# If you create a new virtual environment, you should update the path above and activate yours.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

## force off-line usage of the HF cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

srun torchrun_jsc \
     --nproc_per_node=gpu \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
     --rdzv_backend=c10d \
     "hf_example.py" -t ddp

srun torchrun_jsc \
     --nproc_per_node=gpu \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
     --rdzv_backend=c10d \
     "hf_example.py" -t fsdp
