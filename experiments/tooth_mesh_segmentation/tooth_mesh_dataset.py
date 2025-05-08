import os
import torch
from torch.utils.data import Dataset
import potpourri3d as pp3d
import diffusion_net
import numpy as np

class ToothMeshDataset(Dataset):
    """
    Dataset for 3D tooth mesh segmentation.
    """

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):
        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.n_class = 64  # 牙齿分割有32个类别

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []

        split_file = "train.txt" if train else "test.txt"
        with open(os.path.join(self.root_dir, split_file)) as f:
            files = [line.strip() for line in f]

        batch_iter_flag = True
        if batch_iter_flag:
            for f in files:
                mesh_file = os.path.join(self.root_dir, "meshes", f)
                label_file = os.path.join(self.root_dir, "labels", f[:-4] + ".txt")

                verts, faces = pp3d.read_mesh(mesh_file)
                labels = torch.tensor(np.loadtxt(label_file).astype(int))

                verts = diffusion_net.geometry.normalize_positions(torch.tensor(verts).float())
                faces = torch.tensor(faces)

                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.labels_list.append(labels)
            
            print("Loaded {} meshes".format(len(self.verts_list)))

            self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = \
                diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        else:
            self.frames_list = []
            self.massvec_list = []
            self.L_list = []
            self.evals_list = []
            self.evecs_list = []
            self.gradX_list = []
            self.gradY_list = []
            self.name_list = []

            print("Loading meshes...",len(files))
            for f in files:
                mesh_file = os.path.join(self.root_dir, "meshes", f)
                label_file = os.path.join(self.root_dir, "labels", f[:-4] + ".txt")

                verts, faces = pp3d.read_mesh(mesh_file)
                labels = torch.tensor(np.loadtxt(label_file).astype(int)) + 1

                verts = diffusion_net.geometry.normalize_positions(torch.tensor(verts).float())
                faces = torch.tensor(faces)

                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.labels_list.append(labels)
                self.name_list.append(f)
            
                frames, massvec, L, evals, evecs, gradX, gradY = \
                    diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir, normals=None, overwrite_cache=False)
                self.frames_list.append(frames)
                self.massvec_list.append(massvec)
                self.L_list.append(L)
                self.evals_list.append(evals)
                self.evecs_list.append(evecs)
                self.gradX_list.append(gradX)
                self.gradY_list.append(gradY)


    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        # print("Loading mesh {}...".format(self.name_list[idx]))
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], \
               self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], \
               self.labels_list[idx]
