#!/bin/bash
LOG_FILE=$(date +%Y%m%d_%H_%M_$(hostname).log)

echo "### Training Begin... ###"
python tooth_mesh_segmentation.py \
        --input_feature='hks' \
        --c_width 128 \
        --n_block 6 \
        --train_num 20 \
        --test_num 5 \
        2>&1| tee ${LOG_FILE} &
