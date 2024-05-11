#!/bin/bash

checkpoint_folder="checkpoints"
vit_b16_mae_checkpoint="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"

if [ ! -f "$checkpoint_folder/mae_pretrain_vit_base.pth" ]; then
    wget -P "$checkpoint_folder" "$vit_b16_mae_checkpoint"
fi
