import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import sys
import colorsys
from PIL import Image as PILImage

def gaussian_heat(center, points, sigma=3.0):
    dists = np.linalg.norm(points - center, axis=1)
    heat = np.exp(-0.5 * (dists / sigma) ** 2)
    return heat

def set_view(vis, param):
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array(param["lookat"]))
    ctr.set_front(np.array(param["front"]))
    ctr.set_up(np.array(param["up"]))
    ctr.set_zoom(param["zoom"])

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
        print("exp. python step2_visualize_simulated_fdi_heatmap.py cd # 相机细节截图模式")
        print("exp. python step2_visualize_simulated_fdi_heatmap.py m # 合并视图")
        return
    
    mode = sys.argv[1]
    print(f"运行模式: {mode}")
    
    ply_path = "../visual_dgcnn_vs_hks/Ablation_Experiment1/X2J0T3NG_upper_colored_GT.ply"
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"无法读取三角网格文件: {ply_path}")
        return
    #通过FPS最远点采样 对mesh进行降采样
    
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=100000)  # 可选：简化网格
    
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
        # Use smaller sigma for steeper gradient
        heat = gaussian_heat(center, vertices[mask], sigma=3.5)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        # Apply power function to make transition more dramatic
        heat = heat ** 3  # Square the values to make gradient more pronounced
        color = base_colors[i % len(base_colors)]
        colors[mask] = heat[:, None] * color + (1 - heat)[:, None] * np.ones(3)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if mode == "v":  # 可视化网格，复制相机位姿参数
        # 使用o3d.visualization.Visualizer()可视化mesh，显示三角面片的边
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=1280, height=1080)
        vis.add_geometry(mesh)
        
        # 创建点云来突出显示顶点
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        point_cloud.colors = mesh.vertex_colors
        vis.add_geometry(point_cloud)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.mesh_show_wireframe = True  # 显示三角面片的边
        render_option.line_width = 0.01  # 减小线宽以弱化三角面片边缘
        # render_option.line_color = np.array([0.7, 0.7, 0.7])  # 设置三角面片边框为灰色
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
        render_option.point_size = 4.0  # 增加点大小以突出顶点
        
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        
    elif mode == "c": # 生成完整视图
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=1960, height=1080)
        
        # 突出显示顶点
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        point_cloud.colors = mesh.vertex_colors
        vis.add_geometry(point_cloud)
        point_option = vis.get_render_option()
        point_option.point_size = 2.0  # 增加点大小以突出顶点
        
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # 读取并设置相机参数
        camera_json_path = "camera_view.json"
        if os.path.exists(camera_json_path):
            with open(camera_json_path, "r") as f:
                cam = json.load(f)
                cam_param = cam["trajectory"][0]
                set_view(vis, cam_param)
                vis.poll_events()
                vis.update_renderer()
        else:
            print(f"相机参数文件不存在: {camera_json_path}")

        # 设置渲染选项
        render_option = vis.get_render_option()
        # render_option.light_on = True  # 打开光照
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
        # render_option.mesh_show_wireframe = True  # 显示三角面片的边
        # render_option.line_width = 0.001  # 设置边线宽度更细
        
        # 更新渲染
        vis.poll_events()
        vis.update_renderer()
        
        # 截取当前视图为图片
        vis.capture_screen_image("simulated_fdi_heatmap.png", do_render=True)
        print("已保存截图为 simulated_fdi_heatmap.png")
        vis.destroy_window()
    elif mode == "cd": # 生成细节视图
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=1280, height=1080)
        
        # 添加网格但弱化边缘
        vis.add_geometry(mesh)
        
        # 突出显示顶点 - 放在后面添加以确保顶点在网格上方显示
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        point_cloud.colors = mesh.vertex_colors
        vis.add_geometry(point_cloud)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 4.0  # 增大点大小使顶点更加明显
        render_option.line_width = 0.1  # 减小线宽以弱化三角面片边缘
        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
        render_option.mesh_show_wireframe = True
        render_option.light_on = True
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
        # render_option.mesh_line_color = np.array([0.9, 0.9, 0.9])  # 非常浅的灰色边缘线，更弱化
        
        vis.poll_events()
        vis.update_renderer()

        # 读取并设置相机参数
        camera_json_path = "camera_view_detail.json"
        if os.path.exists(camera_json_path):
            with open(camera_json_path, "r") as f:
                cam = json.load(f)
                cam_param = cam["trajectory"][0]
                set_view(vis, cam_param)
                vis.poll_events()
                vis.update_renderer()
        else:
            print(f"相机参数文件不存在: {camera_json_path}")
        
        # 截取当前视图为图片
        vis.capture_screen_image("simulated_fdi_heatmap_detail.png", do_render=True)
        print("已保存截图为 simulated_fdi_heatmap_detail.png")
        vis.destroy_window()
    elif mode == "m": # 叠加细节视图到主视图
        # 创建一个新的图像，将 detail_image 缩放后叠加到 main_image 上
        detail_image_path = "simulated_fdi_heatmap_detail.png"
        main_image_path = "simulated_fdi_heatmap.png"
        if not os.path.exists(detail_image_path) or not os.path.exists(main_image_path):
            print("请先生成细节视图和主视图")
            return
        detail_image = PILImage.open(detail_image_path)
        main_image = PILImage.open(main_image_path)
        # 缩放细节图像到主图像的1/4大小
        detail_image = detail_image.resize((detail_image.width // 2, detail_image.height // 2), PILImage.LANCZOS)
        # 计算叠加位置：右上角
        position = (main_image.width - detail_image.width, 0)
        # 创建一个新的图像，将 detail_image 粘贴到 main_image 上
        combined_image = PILImage.new("RGB", main_image.size)
        combined_image.paste(main_image, (0, 0))
        combined_image.paste(detail_image, position)
        # 保存合并后的图像
        combined_image.save("simulated_fdi_heatmap_combined.png")
        print("已保存合并后的图像为 simulated_fdi_heatmap_combined.png")
    else:
        print(f"未知模式: {mode}，请使用 'v' 或 'c'")

if __name__ == "__main__":
    main()
