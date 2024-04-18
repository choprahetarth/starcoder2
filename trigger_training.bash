#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4 # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account='bbvz-delta-gpu'
#SBATCH --job-name="finetune/custom_fine_tune.py"
#SBATCH --time=30:00:00
### GPU options ###
#SBATCH --gpus-per-node=4
# SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
# module load python anaconda3_gpu  # ... or any appropriate modules
# module list  # job documentation and metadata
# echo "job is starting on `hostname`"

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate hetarth_py10

# Set the HF_HOME environment variable
export HF_HOME=/projects/bbvz/choprahetarth
export WANDB_DIR=/projects/bbvz/choprahetarth  # replace with your desired path
export WANDB_API_KEY=e1b18fcb1054536d8c6958c02a175ddff40f4914
export HF_API_KEY=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


# chmod +x /u/choprahetarth/all_files/starcoder/finetune/*
srun --account=bbvz-delta-gpu /u/choprahetarth/all_files/starcoder/finetune/environment_setup.py

srun --account=bbvz-delta-gpu \
python3 -m torch.distributed.run \
--nproc_per_node=4 \
finetune/custom_fine_tune.py \
--model_id="bigcode/starcoder2-3b" \
--subset="data/finetune" \
--split="train" \
--streaming \
--dataset_name="/u/choprahetarth/all_files/data/train_ftdata-new-small.json" \
--size_valid_set 1525 \
--seq_length 512 \
--max_steps 214 \
--batch_size 16 \
--input_column_name="input" \
--output_column_name="output" \
--gradient_accumulation_steps 2 \
--learning_rate 1e-4 \
--lr_scheduler_type="cosine" \
--num_warmup_steps 2 \
--weight_decay 0.05 \
--output_dir="/projects/bbvz/choprahetarth/new_experiments/experiment_2" \
--seed 1234 \
--save_freq 10


## So the total dataset is 15246 rows
## and the total batch size across all GPUs is 256 (64*4 (gpus))
## So one epoch is (13721/256) approx 60 batches
## and ten percent of the dataset is 60*0.1 approx 6 batches