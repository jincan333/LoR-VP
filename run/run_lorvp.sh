#!/bin/bash
prefix='LoR_VP'
network='vit_b_16_21k'
dataset='tiny_imagenet'
# resnet50_miil_21k.pth   ViT-B_16.npz   ViT-L_16.npz   official
pretrain_path='../model/resnet50_miil_21k.pth'
input_size=224
output_size=224
downstream_mapping='lp'
mapping_freq=1
prompt_method='lor_vp'
bar_width=4
init_method='zero,normal'
train_batch_size=256
randomcrop=0
optimizer='sgd'
scheduler='cosine'
lr=0.02
weight_decay=0.0001
epochs=20
gpu=0
seed=1234
eval_frequency=1


experiment_name=${network}_${dataset}_is${input_size}_os${output_size}_${downstream_mapping}_freq${mapping_freq}_${prompt_method}_bwid${bar_width}_${init_method}_btz${train_batch_size}_randc${randomcrop}_${optimizer}_${scheduler}_lr${lr}_epochs${epochs}_wd${weight_decay}_gpu${gpu}_seed${seed}
log_folder_name=logs/${prefix}_${dataset}
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi

log_filename=${log_folder_name}/${experiment_name}.log
while true; do
    experiment_name=${network}_${dataset}_is${input_size}_os${output_size}_${downstream_mapping}_freq${mapping_freq}_${prompt_method}_bwid${bar_width}_${init_method}_btz${train_batch_size}_randc${randomcrop}_${optimizer}_${scheduler}_lr${lr}_epochs${epochs}_wd${weight_decay}_gpu${gpu}_seed${seed}
    log_filename=${log_folder_name}/${experiment_name}.log
    if [ ! -f "${log_filename}" ]; then
        break
    fi
    seed=$((seed + 1))
done

save_path=/research/cbim/medical/cj574/visual_prompt/ckpt/${prefix}_${dataset}/${experiment_name}
if [ ! -d ${save_path} ]; then
    mkdir -p ${save_path}
fi

nohup python trainer/image_classification_ddp.py \
    --exp_name ${experiment_name} \
    --save_path ${save_path} \
    --network ${network} \
    --dataset ${dataset} \
    --pretrain_path ${pretrain_path} \
    --input_size ${input_size} \
    --output_size ${output_size} \
    --downstream_mapping ${downstream_mapping} \
    --mapping_freq ${mapping_freq} \
    --prompt_method ${prompt_method} \
    --bar_width ${bar_width} \
    --init_method ${init_method} \
    --train_batch_size ${train_batch_size} \
    --randomcrop ${randomcrop} \
    --optimizer ${optimizer} \
    --scheduler ${scheduler} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    --epochs ${epochs} \
    --gpu ${gpu} \
    --seed ${seed} \
    --eval_frequency ${eval_frequency} \
> ${log_filename} 2>&1 &