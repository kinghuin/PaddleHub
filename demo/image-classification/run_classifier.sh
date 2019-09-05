#!/usr/bin/env bash
export FLAGS_eager_delete_tensor_gb=0.0
python -u download.py && \

DATASET0="flowers"
DATASET1="dogcat"
DATASET2="indoor67"
DATASET3="food101"
DATASET4="stanforddogs"



GPU=1,2
batch_size=40



####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0

DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

####*************
####*************
####*************
warmup_proportion=0.01
start_point=${warmup_proportion}
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0

DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}


####*************
####*************
####*************
warmup_proportion=0.05
start_point=${warmup_proportion}
end_learning_rate=0.03125e-4
dis_blocks=0
frz_blocks=0
cut_fraction=0

DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET4}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}







####*************
####*************
####*************
warmup_proportion=0.1
start_point=${warmup_proportion}
end_learning_rate=0.03125e-4
dis_blocks=0
frz_blocks=0
cut_fraction=0


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}





####*************
####*************
####*************
warmup_proportion=0.2
start_point=${warmup_proportion}
end_learning_rate=0.03125e-4
dis_blocks=0
frz_blocks=0
cut_fraction=0


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}




####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0.01


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}




####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0.05


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}


DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}



####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0.1


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}


DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}



####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=0
frz_blocks=0
cut_fraction=0.2

DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}




####*************
####*************
####*************
warmup_proportion=0
start_point=1
end_learning_rate=0
dis_blocks=3
num_epoch=1
frz_blocks=0
cut_fraction=0


DATASET=${DATASET1}   #HERE
sample=True
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET1}   #HERE
sample=False
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --sample=${sample}

DATASET=${DATASET2}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --num_epoch=${num_epoch}

DATASET=${DATASET3}   #HERE
export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --num_epoch=${num_epoch}


DATASET=${DATASET4}   #HERE

export CUDA_VISIBLE_DEVICES=${GPU}
CKPT_DIR="./ckpt_${DATASET}"
python -u img_classifier.py \
        --checkpoint_dir=${CKPT_DIR} \
        --dataset=${DATASET} \
        --warmup_proportion=${warmup_proportion} \
        --start_point=${start_point} \
        --end_learning_rate=${end_learning_rate} \
        --dis_blocks=${dis_blocks} \
        --frz_blocks=${frz_blocks} \
        --cut_fraction=${cut_fraction} \
        --batch_size=${batch_size} \
        --num_epoch=${num_epoch} \
        --sample=${sample}
