#!/bin/bash
LOG_FILE=$(date +%Y%m%d_%H_%M_$(hostname).log)

echo "### Training Begin... ###"
python tooth_mesh_segmentation.py \
        --input_feature='hks' \
        --C_width 128 \
        --N_block 6 \
        2>&1| tee ${LOG_FILE} &
