import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import sys

def gaussian_heat(center, points, sigma=3.0):
    dists = np.linalg.norm(points - center, axis=1)
    heat = np.exp(-0.5 * (dists / sigma) ** 2)
    return heat

def set_view_from_viewtrajectory(vis, viewtrajectory_json):
    ctr = vis.get_view_control()
    traj = viewtrajectory_json["trajectory"][0]
    ctr.set_lookat(np.array(traj["lookat"]))
    ctr.set_front(np.array(traj["front"]))
    ctr.set_up(np.array(traj["up"]))
    ctr.set_zoom(traj["zoom"])

def load_centers_and_colors(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    centers = [item[1] for item in data]
    base_colors = [item[0] for item in data]
    return np.array(centers), np.array(base_colors)

def main():
    
    if len(sys.argv) < 2:
        print("python step2_visualize_simulated_fdi_heatmap.py <mode>")
        print("exp. python step2_visualize_simulated_fdi_heatmap.py v # 可视化模式")
        print("exp. python step2_visualize_simulated_fdi_heatmap.py c # 相机截图模式")
        return
    
    mode = sys.argv[1]
    print(f"运行模式: {mode}")
    
    ply_path = "../visual_dgcnn_vs_hks/Ablation_Experiment1/X2J0T3NG_upper_colored_GT.ply"
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

    # 用每个顶点最近的center分组
    dists = np.linalg.norm(vertices[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    colors = np.zeros((vertices.shape[0], 3))
    for i, center in enumerate(centers):
        mask = (labels == i)
        heat = gaussian_heat(center, vertices[mask], sigma=3.0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        color = base_colors[i % len(base_colors)]
        colors[mask] = heat[:, None] * color + (1 - heat)[:, None] * np.ones(3)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if mode == "v":
       o3d.visualization.draw_geometries([mesh], window_name="牙位号渐变热力图")
    elif mode == "c":
        # 使用Open3D Visualizer设置相机并截图
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=1280, height=1080)
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # 读取并设置相机参数
        camera_json_path = "camera_view.json"
        try:
            with open(camera_json_path, "r") as f:
                camera_params = json.load(f)
            set_view_from_viewtrajectory(vis, camera_params)
            vis.poll_events()
            vis.update_renderer()
        except Exception as e:
            print("未设置相机参数或设置失败：", e)

        # 截取当前视图为图片
        vis.capture_screen_image("simulated_fdi_heatmap.png", do_render=True)
        print("已保存截图为 simulated_fdi_heatmap.png")
        vis.destroy_window()
    else:
        print(f"未知模式: {mode}，请使用 'v' 或 'c'")

if __name__ == "__main__":
    main()
