import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from tooth_mesh_dataset import ToothMeshDataset
import random
import getpass

fdi_colors_map = {
    0: [228, 228, 228],
    11: [255, 102, 102],
    12: [255, 217, 102],
    13: [179, 255, 102],
    14: [102, 255, 140],
    15: [102, 255, 255],
    16: [102, 140, 255],
    17: [179, 102, 255],
    18: [255, 102, 217],
    21: [255, 153, 153],
    22: [255, 230, 153],
    23: [204, 255, 153],
    24: [153, 255, 179],
    25: [153, 255, 255],
    26: [153, 179, 255],
    27: [204, 153, 255],
    28: [255, 153, 230],
    31: [255, 153, 153],
    32: [255, 230, 153],
    33: [204, 255, 153],
    34: [153, 255, 179],
    35: [153, 255, 255],
    36: [153, 179, 255],
    37: [204, 153, 255],
    38: [255, 153, 230],
    41: [255, 102, 102],
    42: [255, 217, 102],
    43: [179, 255, 102],
    44: [102, 255, 140],
    45: [102, 255, 255],
    46: [102, 140, 255],
    47: [179, 102, 255],
    48: [255, 102, 217],
}
 
def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        # Evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


# Do an evaluation pass on the test dataset
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for i, data in enumerate(tqdm(test_loader)):

            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

            # Save the results as a .ply file
            verts_np = verts.cpu().numpy()
            faces_np = faces.cpu().numpy()
            pred_labels_np = pred_labels.cpu().numpy()

            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts_np)
            mesh.triangles = o3d.utility.Vector3iVector(faces_np)

            colors = []
            for label in pred_labels_np:
                # Use fdi_colors_map to assign colors, default to black if label not found
                colors.append(np.array(fdi_colors_map.get(label, [0, 0, 0])) / 255.0)
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

            # Save to .ply
            if not train:
                output_path = os.path.join(base_path, f"test_out/test_output_{i}.ply")
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Saved visualization to {output_path}")


    test_acc = correct / total_num
    return test_acc 


# if train:
#     print("Training...")

#     for epoch in range(n_epoch):
#         train_acc = train_epoch(epoch)
#         test_acc = test()
#         print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

#     print(" ==> saving last model to " + model_save_path)
#     torch.save(model.state_dict(), model_save_path)


def ramdom_dataset_file(dataset_path, train_num=30, test_num=10):
    train_txt_file = os.path.join(dataset_path, "train-bak.txt")
    test_txt_file = os.path.join(dataset_path, "test-bak.txt")
    # 读取 train-bak.txt 文件的每一行，随机选择其中 10 行写入 train.txt
    with open(train_txt_file, 'r') as f:
        lines = f.readlines()
    selected_lines = random.sample(lines, train_num)  # 随机选择 10 行
    with open(os.path.join(dataset_path, "train.txt"), 'w') as f:
        f.writelines(selected_lines)
    print("### Train.txt (first 5 lines):")
    print("\n".join(line.strip() for line in selected_lines[:5]))
    
     # 读取 test-bak.txt 文件的每一行，随机选择其中 10 行写入 test.txt
    with open(test_txt_file, 'r') as f:
        lines = f.readlines()
    selected_lines = random.sample(lines, test_num)  # 随机选择 5 行
    with open(os.path.join(dataset_path, "test.txt"), 'w') as f:
        f.writelines(selected_lines)
    print("### Test.txt (first 5 lines):")
    print("\n".join(line.strip() for line in selected_lines[:5]))


if __name__ == "__main__":
    # === Options
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
    parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default='xyz')
    parser.add_argument("--C_width", type=int, default=128)
    parser.add_argument("--N_block", type=int, default=4)
    args = parser.parse_args()

    # System settings
    device = torch.device('cuda:0')
    dtype = torch.float32

    # Problem settings
    n_class = 64  # 牙齿分割有10个类别
    input_features = args.input_features
    k_eig = 128

    # Training settings
    train = not args.evaluate
    n_epoch = 500
    lr = 1e-3
    decay_every = 10
    decay_rate = 0.8
    augment_random_rotate = (input_features == 'xyz')
    c_width = args.C_width
    n_block = args.N_block

    # Paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, "data", "op_cache")
    save_dir = os.path.join(base_path, "data", "saved_models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    pretrain_path = os.path.join(save_dir, "tooth_mesh_seg_{}_{}x{}.pth".format(input_features, c_width, n_block))
    model_save_path = os.path.join(save_dir, "tooth_mesh_seg_{}_{}x{}.pth".format(input_features, c_width, n_block))
    username = getpass.getuser()
    dataset_path = os.path.join(base_path, f"/home/{username}/Datasets/Teeth3DS+_dataset")
    
    print("Dataset path: ", dataset_path)

    # === Load datasets

    # test_dataset = ToothMeshDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    # test_loader = DataLoader(test_dataset, batch_size=None)

    # if train:
    #     train_dataset = ToothMeshDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    #     print("Train dataset size: ", len(train_dataset))
    #     train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    # === Create the model

    C_in = {'xyz': 3, 'hks': 16}[input_features]

    model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                            C_out=n_class,
                                            C_width=c_width,
                                            N_block=n_block,
                                            last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                            outputs_at='vertices',
                                            dropout=True)

    model = model.to(device)

    if not train:
        print("Loading pretrained model from: " + str(pretrain_path))
        model.load_state_dict(torch.load(pretrain_path))

    # === Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
    if train:
        print("Training...")

        # 检查是否存在之前保存的模型
        if os.path.exists(model_save_path):
            print(f"Loading existing model from: {model_save_path}")
            model.load_state_dict(torch.load(model_save_path))
            print("Model loaded. Resuming training...")

        test_acc_ref = 0
        for epoch in range(n_epoch):
            ramdom_dataset_file(dataset_path)
            test_dataset = ToothMeshDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
            test_loader = DataLoader(test_dataset, batch_size=None)
            print("Test dataset size: ", len(test_loader))
            train_dataset = ToothMeshDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
            train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
            print("Train dataset size: ", len(train_dataset))
            
            train_acc = train_epoch(epoch)
            test_acc = test()
            print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.2f}%".format(epoch, 100*train_acc, 100*test_acc))
            if test_acc > test_acc_ref:
                test_acc_ref = test_acc
                model_tmp_save_path = os.path.join(base_path, \
                    "data/saved_models/tooth_mesh_seg_{}_6x128_best_epoch{}_acc{:06.3f}.pth".format(input_features,epoch,100*test_acc))
                print(" ==> saving best model to " + model_tmp_save_path)
                torch.save(model.state_dict(), model_tmp_save_path)

        print(" ==> saving last model to " + model_save_path)
        torch.save(model.state_dict(), model_save_path)
    # Test
    else:
        test_old_file = os.path.join(dataset_path, "test-bak.txt")
        test_new_file = os.path.join(dataset_path, "test.txt")
        os.system("cp " + test_old_file + " " + test_new_file)
        test_dataset = ToothMeshDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
        test_loader = DataLoader(test_dataset, batch_size=None)

        test_acc = test()
        print("Overall test accuracy: {:06.3f}%".format(100*test_acc))