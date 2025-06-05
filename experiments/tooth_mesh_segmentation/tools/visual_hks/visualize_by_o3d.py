import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

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
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=880, height=720)
    vis.add_geometry(mesh)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array(bg_color)
    render_option.light_on = True
    render_option.mesh_show_back_face = True
  
    # 设置视角
    view_control = vis.get_view_control()

    view_control.set_front([-0.22153177751855729, -0.15593230743848191, -0.96260520830004837])
    view_control.set_lookat([-0.33491428725845185, 0.00015801633380072011, 9.7629913444372072])
    view_control.set_up([-0.0053271844783358896, -0.98692408539105614, 0.16109770569614315])

    # 调整缩放以适应模型 - 增大缩放使模型更大
    view_control.set_zoom(0.4)  # 增加缩放比例，从0.8增加到1.2
    
    # 居中模型
    vis.get_view_control().set_lookat(mesh.get_center())
    
    # 更新渲染
    vis.update_renderer()
    
    # 捕获图像
    image = vis.capture_screen_float_buffer(do_render=True)
    img_array = np.asarray(image)
    
    # 确保图像值在0-1范围内，修复浮点RGB值错误
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

def main():
    # # 解析命令行参数
    # import argparse
    # parser = argparse.ArgumentParser(description='使用Open3D加载PLY文件并截取二维视图')
    # parser.add_argument('ply_file', help='要加载的PLY文件路径')
    # args = parser.parse_args()
    
    # 分别打印每个时间尺度的PLY视角图片
    for i in range(0, 4):
        ply_path = "hks_enhanced_t" + str(i) + ".ply"
        print(f"处理文件: {ply_path}")
        if not os.path.exists(ply_path):
            print(f"指定的文件不存在，使用默认文件: {ply_path}")
        capture_2d_view(ply_path)

    time_steps = [0.001, 0.1, 10.0, 1000.0]  # 跨度更大的对数尺度

    # 使用matplotlib 建立二维图片坐标系，依次读入4张png图片，横向排布，每个图像下方标注time_steps值，最后保存为一张整图
    images = []
    for i in range(0, 4):
        img_path = "render_hks_enhanced_t" + str(i) + ".png"
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            images.append(img)
        else:
            print(f"警告: 图片文件 '{img_path}' 不存在，跳过该文件。")
    if not images:
        print("错误: 没有找到任何有效的图片文件。")
        return
    # 创建一个新的图形
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]
    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴显示
        # 在图片下方添加time_steps标签
        ax.set_title(f"t = {time_steps[idx]}", fontsize=16, pad=20)
    # 保存合并后的图片
    combined_image_path = "combined_hks_visualization.png"
    plt.savefig(combined_image_path, bbox_inches='tight', dpi=800)
    print(f"已保存合并后的图片: {combined_image_path}")
    plt.close(fig)  # 关闭图形以释放内存
    
        
if __name__ == "__main__":
    main()
