import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import json


def capture_2d_view(ply_path, bg_color=[1.0, 1.0, 1.0]):
    """
    加载PLY文件并截取二维视图
    
    参数:
    - ply_path: PLY文件路径
    - output_dir: 输出目录，若为None则使用当前目录
    - bg_color: 背景颜色，默认为白色
    """
    # 检查文件是否存在
    if not os.path.exists(ply_path):
        print(f"错误: 文件 '{ply_path}' 不存在")
        return False
    
    # 加载PLY文件
    print(f"加载PLY文件: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        print(f"原始顶点颜色数量: {len(colors)}")
        
        # 找到白色顶点（RGB值都为1.0）
        white_mask = np.all(colors == [1.0, 1.0, 1.0], axis=1)
        white_count = np.sum(white_mask)
        print(f"发现 {white_count} 个白色顶点")
        
        # 将白色顶点设置为灰色
        colors[white_mask] = [0.75, 0.75, 0.75]
        
        # 更新网格的顶点颜色
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        print(f"已将 {white_count} 个白色顶点改为灰色") 
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=1080)
    vis.add_geometry(mesh)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array(bg_color)
    render_option.light_on = True
    render_option.mesh_show_back_face = True
    
    # 设置视角
    view_control = vis.get_view_control()

    # 从cam.json文件中获取相机参数
    with open("cam.json", "r") as f:
        cam_params = json.load(f)
    view_control.set_front(cam_params["trajectory"][0]["front"])
    view_control.set_lookat(cam_params["trajectory"][0]["lookat"])
    view_control.set_up(cam_params["trajectory"][0]["up"])
    view_control.set_zoom(cam_params["trajectory"][0]["zoom"])
                                
    # view_control.set_front([0.095319725921393136, -0.35295657480940745, -0.93077161868477798 ])
    # view_control.set_lookat([ -1.2317, 8.4314499999999981, 13.956545 ])
    # view_control.set_up([0.086416494734743679, -0.92856506188294063, 0.36096968749202596])
    # 调整缩放以适应模型 - 增大缩放使模型更大
    # view_control.set_zoom(0.4)  # 增加缩放比例，从0.8增加到1.2
    
    # 居中模型
    # vis.get_view_control().set_lookat(mesh.get_center())
    
    # 更新渲染
    vis.update_renderer()
    
    # 捕获图像
    image = vis.capture_screen_float_buffer(do_render=True)
    img_array = np.asarray(image)
    
    # # 确保图像值在0-1范围内，修复浮点RGB值错误
    if np.max(img_array) > 1.0 or np.min(img_array) < 0.0:
        img_array = np.clip(img_array, 0.0, 1.0)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_basename = os.path.splitext(os.path.basename(ply_path))[0]
    output_file = os.path.join("render_" + f"{file_basename}.png")
    
    # 保存图像
    plt.imsave(output_file, img_array, dpi=600)
    print(f"已保存二维视图: {output_file}")
    
    # 不关闭可视化器，保持窗口可交互
    # vis.run()  # 允许用户通过鼠标交互旋转、缩放等, 获取视角参数
    vis.destroy_window()
    
    return True

def visual_mesh(mesh_path):
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # 遍历网格顶点，将白色[1.0,1.0,1.0]设置为灰色[0.75, 0.75, 0.75]
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        print(f"原始顶点颜色数量: {len(colors)}")
        
        # 找到白色顶点（RGB值都为1.0）
        white_mask = np.all(colors == [1.0, 1.0, 1.0], axis=1)
        white_count = np.sum(white_mask)
        print(f"发现 {white_count} 个白色顶点")
        
        # 将白色顶点设置为灰色
        colors[white_mask] = [0.75, 0.75, 0.75]
        
        # 更新网格的顶点颜色
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        print(f"已将 {white_count} 个白色顶点改为灰色")
    
    # # 如果网格没有颜色信息，设置一个较暗的颜色来减少曝光
    # if not mesh.has_vertex_colors():
    #     mesh.paint_uniform_color([1, 1, 1])  # 中等灰色，避免过亮
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=1080)
    vis.add_geometry(mesh)
    
    # 设置渲染选项和光照
    render_option = vis.get_render_option()
    # render_option.background_color = np.array([0.3, 0.3, 0.3])  # 中等灰色背景
    render_option.light_on = True  
    render_option.mesh_show_back_face = True  # 关闭背面显示减少反光
    
    # 移除不支持的SmoothShade选项，使用默认着色
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.SmoothShade
    
    # 设置基本渲染选项来实现光顺效果
    render_option.point_size = 0
    render_option.line_width = 0
    render_option.point_show_normal = False  # 关闭法线显示
    render_option.show_coordinate_frame = False  # 关闭坐标轴
    render_option.mesh_show_wireframe = False  # 确保不显示线框
    
    vis.run()
    
def main():
    
    if len(sys.argv) < 2:
        print("用法: python step4_take_pic_by_cam_parm.py <ply_file> <type>")
        return
    
    type = 0
    if len(sys.argv) == 2:
        mesh_path = sys.argv[1]
        type = 1
    elif len(sys.argv) == 3:
        mesh_path = sys.argv[1]
        type = int(sys.argv[2])

    print("type:", type)
        
    if type == 1:
        visual_mesh(mesh_path)
    elif type == 2:
        capture_2d_view(mesh_path)
    else:
        print("错误: 未知的类型参数 {type}, 请使用 1 或 2。")
        return
        
if __name__ == "__main__":
    main()
