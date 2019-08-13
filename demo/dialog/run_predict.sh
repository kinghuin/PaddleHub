export FLAGS_eager_delete_tensor_gb=0.0
# export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc and so on for different task
DATASET="udc"
CKPT_DIR="./ckpt_${DATASET}_210"
# Recommending hyper parameters for difference task

python -u predict.py --checkpoint_dir=$CKPT_DIR \
                            --max_seq_len=128 \
                            --use_gpu=True \
                            --dataset=${DATASET} \
                            --batch_size=150 \
