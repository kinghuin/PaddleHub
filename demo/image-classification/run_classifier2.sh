#!/usr/bin/env bash
export FLAGS_eager_delete_tensor_gb=0.0
python -u download.py && \

DATASET0="flowers"
DATASET1="dogcat"
DATASET2="indoor67"
DATASET3="food101"


GPU=2,3,4,5,6


####*************
####*************
####*************
warmup_proportion=0.1
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}



####*************
####*************
####*************
warmup_proportion=0.2
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}



####*************
####*************
####*************
warmup_proportion=0.0
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0.01

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}





####*************
####*************
####*************
warmup_proportion=0.0
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0.05

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET0}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET1}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}






####*************
####*************
####*************
warmup_proportion=0.0
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0.1

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}


DATASET=${DATASET0}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}

DATASET=${DATASET1}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}




####*************
####*************
####*************
warmup_proportion=0.0
end_learning_rate=0.0
dis_blocks=0
frz_blocks=0
cut_fraction=0.2

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}


DATASET=${DATASET1}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction}
