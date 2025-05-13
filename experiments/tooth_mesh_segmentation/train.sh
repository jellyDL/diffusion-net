#!/bin/bash
LOG_FILE=$(date +%Y%m%d_%H_%M_$(hostname).log)

echo "### Training Begin... ###"
python tooth_mesh_segmentation.py \
        --input_feature='hks' \
        2>&1| tee ${LOG_FILE} &
