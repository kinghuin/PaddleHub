export FLAGS_eager_delete_tensor_gb=0.0
# export CUDA_VISIBLE_DEVICES=0

DATASET="udc"
CKPT_DIR="./ckpt_${DATASET}_210_duiqi"
# Recommending hyper parameters for difference task
# UDC: batch_size=16, weight_decay=0.01, num_epoch=2, max_seq_len=210, lr=2e-5

python -u dialog.py \
                   --batch_size=6720 \
                   --use_gpu=True \
                   --dataset=${DATASET} \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=2e-5 \
                   --weight_decay=0.01 \
                   --max_seq_len=210 \
                   --num_epoch=2 \
                   --use_pyreader=True \
                   --use_data_parallel=True \
                   --warmup_proportion=0.1 \
                   --in_tokens=True \
                   --log_interval=20 \
                   --eval_interval=1000 \
