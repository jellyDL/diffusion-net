### Tooth Mesh Segmentation

This experiment performs instance segmentation on 3D tooth meshes.

### Data

Place the dataset in the `Teeth3DS+_dataset` directory. The dataset should include:
- `meshes/`: Directory containing `.obj` files for the meshes.
- `labels/`: Directory containing `.txt` files for per-vertex labels.
- `train.txt`: List of training mesh filenames.
- `test.txt`: List of testing mesh filenames.

### Training

To train the model, run:
```bash
python tooth_mesh_segmentation.py --input_features=xyz
```

To use heat kernel signature (HKS) features:
```bash
python tooth_mesh_segmentation.py --input_features=hks
```

### Evaluation

To evaluate a pretrained model, run:
```bash
python tooth_mesh_segmentation.py --evaluate --input_features=xyz
```
