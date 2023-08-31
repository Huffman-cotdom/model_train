export OUTPUT_DIR=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/exhibition_hall/seller/model/seller_multi_label_bz8_0822
python3 run_classification.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --problem_type=multi_label_classification \
    --model_checkpoint=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/models/chinese-macbert-base \
    --train_file=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/exhibition_hall/seller/data/d0822/train_d0822.json \
    --dev_file=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/exhibition_hall/seller/data/d0822/eval_d0822.json \
    --test_file=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/exhibition_hall/seller/data/d0822/test_d0822.json \
    --label_file=/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/exhibition_hall/seller/data/d0822/label2id.json \
    --max_seq_length=256 \
    --do_train \
    --learning_rate=1e-5 \
    --num_train_epochs=30 \
    --batch_size=8 \
    --warmup_proportion=0.1 \
    --weight_decay=0.01 \
    --seed=0 \
    --stop_epoch=2 \
    --loss_type=BCE