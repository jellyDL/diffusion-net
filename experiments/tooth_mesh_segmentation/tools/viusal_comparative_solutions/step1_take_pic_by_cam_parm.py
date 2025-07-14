import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import json
from PIL import Image, ImageDraw, ImageFont

def draw_on_image(image_file, draw_path):
    # 解析 draw.json 文件
    if not os.path.exists(draw_path):
        print(f"错误: 找不到标记文件 {draw_path}")
        return  
    with open(draw_path, 'r') as f:
        draw_data = json.load(f)
        print(f"读取标记数据: {draw_data}")
    
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    width, height = img.size
        
    if draw_data["ellipses"]:
        items = draw_data["ellipses"]
        for item in items:
            boundingbox_W = int(item["boundingbox_W"])
            boundingbox_H = int(item["boundingbox_H"])
            boundingbox_X = int(item["boundingbox_X"])
            boundingbox_Y = int(item["boundingbox_Y"])
            boundingbox_Rotate = int(item["boundingbox_Rotate"])
            boundingbox_LineW = int(item["boundingbox_LineW"])

            left = boundingbox_X - int(boundingbox_W / 2)
            right = boundingbox_X + int(boundingbox_W / 2)
            top = boundingbox_Y - int(boundingbox_H / 2)
            bottom = boundingbox_Y + int(boundingbox_H / 2)
    
            # ellipse_bbox = (left, top, right, bottom)，表示椭圆的外接矩形坐标
            ellipse_bbox = (left, top, right, bottom)
    
            # 计算椭圆中心点
            center_x = boundingbox_X
            center_y = boundingbox_Y
    
            # 创建一个新图像用于旋转
            transparent = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_transparent = ImageDraw.Draw(transparent)
            
            # 在透明层上绘制虚线椭圆
            for angle in range(0, 360, 10):  # Draw dash every 10 degrees
                start_angle = angle
                end_angle = angle + 5  # 5 degrees per dash
                draw_transparent.arc(ellipse_bbox, start=start_angle, end=end_angle, 
                                    fill="red", width=boundingbox_LineW)
            
            # 旋转透明层 (顺时针旋转30度，所以用-30)
            rotated = transparent.rotate(boundingbox_Rotate, 
                                         center=(center_x, center_y), expand=False)
            
            # 将旋转后的椭圆合并到原图
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, rotated)
            img = img.convert('RGB')  # 转回RGB模式保存
            
            img.save(image_file)
            print(f"已在图像上添加旋转30度的虚线椭圆标记: {image_file}") 
    
def capture_2d_view(ply_path, cam_path, output_file, draw_path=None, bg_color=[1.0, 1.0, 1.0]):
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
        colors[white_mask] = [0.6, 0.6, 0.6]
        
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
    with open(cam_path, "r") as f:
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
    
    # 保存图像
    plt.imsave(output_file, img_array, dpi=600)
    print(f"已保存二维视图: {output_file}")
    
    # 不关闭可视化器，保持窗口可交互
    # vis.run()  # 允许用户通过鼠标交互旋转、缩放等, 获取视角参数
    vis.destroy_window()
    
    draw_on_image(output_file, draw_path)
    
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
        colors[white_mask] = [0.6, 0.6, 0.6]
        
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
        print("用法1: python step1_take_pic_by_cam_parm.py <ply_file> 1")
        print("用法2: python step1_take_pic_by_cam_parm.py <ply_file> 2 <cam_path>")
        print("用法3: python step1_take_pic_by_cam_parm.py <ply_dir> 3")
        return
    
    type = 0
    cam_path = "cam.json"
    if len(sys.argv) == 2:
        mesh_path = sys.argv[1]
        type = 1
    elif len(sys.argv) == 3:
        mesh_path = sys.argv[1]
        type = int(sys.argv[2])
    elif len(sys.argv) == 4:
        mesh_path = sys.argv[1]
        type = int(sys.argv[2])
        cam_path = sys.argv[3]

    print("type:", type)
        
    if type == 1:
        visual_mesh(mesh_path)
    elif type == 2:
        file_basename = os.path.splitext(os.path.basename(mesh_path))[0]
        output_file = os.path.join("render_" + f"{file_basename}.png")
        capture_2d_view(mesh_path, cam_path, output_file)
    elif type == 3:
        mesh_dir = mesh_path
        out_dir = "render_pics_"+ mesh_dir.split("_")[-1]
        cam_path = os.path.join(mesh_dir, "cam.json")
        draw_path = os.path.join(mesh_dir, "draw.json")  
        mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply')]
        for mesh_file in mesh_files:
            mesh_path = os.path.join(mesh_dir, mesh_file)
            print(f"Processing {mesh_path} ...")
            output_file = os.path.join(out_dir, "render_" + f"{mesh_file}.png")
            capture_2d_view(mesh_path, cam_path, output_file, draw_path)
    else:
        print("错误: 未知的类型参数 {type}, 请使用 1 或 2。")
        return
        
if __name__ == "__main__":
    main()
