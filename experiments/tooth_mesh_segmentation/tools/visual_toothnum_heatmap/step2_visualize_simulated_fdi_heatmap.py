import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import json

def gaussian_heat(center, points, sigma=3.0):
    dists = np.linalg.norm(points - center, axis=1)
    heat = np.exp(-0.5 * (dists / sigma) ** 2)
    return heat

def load_centers_and_colors(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    centers = [item[1] for item in data]
    base_colors = [item[0] for item in data]
    return np.array(centers), np.array(base_colors)

def main():
    ply_path = "Ablation_Experiment1/X2J0T3NG_upper_colored_GT.ply"
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"无法读取三角网格文件: {ply_path}")
        return
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)

    # 从 color_centers.json 读取 centers 和 base_colors
    centers, base_colors = load_centers_and_colors("color_centers.json")
    print("centers", centers)
    print("base_colors", base_colors)

    # 重新聚类已不需要，直接用centers和base_colors
    # n_teeth = 8
    # kmeans = KMeans(n_clusters=n_teeth, random_state=0).fit(vertices)
    # centers = kmeans.cluster_centers_
    # labels = kmeans.labels_

    # 用每个顶点最近的center分组
    dists = np.linalg.norm(vertices[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    colors = np.zeros((vertices.shape[0], 3))
    for i, center in enumerate(centers):
        mask = (labels == i)
        heat = gaussian_heat(center, vertices[mask], sigma=5.0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        color = base_colors[i % len(base_colors)]
        colors[mask] = heat[:, None] * color + (1 - heat)[:, None] * np.ones(3)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh], window_name="牙位号渐变热力图")

if __name__ == "__main__":
    main()
