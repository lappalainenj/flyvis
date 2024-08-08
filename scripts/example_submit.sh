# train an ensemble of 50 models
python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_training \
        --launch_validation \
        description='test' \
        train=true \
        +delete_if_exists=true

# validate an ensemble of 50 models
 python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_validation

# compute and store synthetic data
python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 0000 \
        --task_name flow \
        --launch_synthetic_recordings


python train_ensemble.py \
        --start 0 \
        --end 50 \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --launch_synthetic_recordings
