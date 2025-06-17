import open3d as o3d
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def set_view(vis, param):
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array(param["lookat"]))
    ctr.set_front(np.array(param["front"]))
    ctr.set_up(np.array(param["up"]))
    ctr.set_zoom(param["zoom"])

def run_view_mode(ply_path):
    print(f"正在读取PLY文件: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"无法读取三角网格文件: {ply_path}")
        return
    mesh.compute_vertex_normals()

    edge_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edge_lines.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]] * len(edge_lines.lines))  # 黑色边缘

    # draw_geometries 不支持 line_width 参数，使用 mesh_show_wireframe=True 以细线显示边缘
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="PLY三角面片及边缘细节",
        mesh_show_back_face=True,
        mesh_show_wireframe=True
    )
    # 如需显示更细的边线，可只显示LineSet对象（但无法与mesh颜色叠加），否则只能用默认线宽
    
    
def capture_view(ply_file, camera_json_path, outfile="mesh_view.png"):
    
    print(f"正在捕获视图: {ply_file}")
    mesh = o3d.io.read_triangle_mesh(ply_file)
    if mesh.is_empty():
        print(f"无法读取三角网格文件: {ply_file}")
        return
    mesh.compute_vertex_normals()

    edge_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edge_lines.colors = o3d.utility.Vector3dVector(
        [[0, 0, 0]] * len(edge_lines.lines))

    with open(camera_json_path, 'r') as f:
        cam = json.load(f)
    cam_param = cam["trajectory"][0]

    vis = o3d.visualization.Visualizer()
    # Set window size for the capture (default is typically 1920x1080)
    width, height = 1080, 1080
    vis.create_window(window_name="Open3D", width=width, height=height, visible=True)
    vis.add_geometry(mesh)
    vis.add_geometry(edge_lines)
    vis.poll_events()
    vis.update_renderer()

    set_view(vis, cam_param)
    vis.poll_events()
    vis.update_renderer()

    # Capture the image
    # Configure rendering for higher quality
    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    render_option.line_width = 0.5
    render_option.mesh_show_wireframe = True
    render_option.mesh_show_back_face = True
    render_option.light_on = True
    
    # Ensure scene is fully rendered before capture
    vis.poll_events()
    vis.update_renderer()
    
    # Capture high resolution image
    vis.capture_screen_image(outfile, do_render=True)
    vis.destroy_window()
    print(f"已保存视图截图为 mesh_view.png (分辨率: {width}x{height})")
    
def run_capture_mode(ply_path, camera_json_path):
    
    # 遍历 ply_path 中的所有 PLY 文件
    for file in os.listdir(ply_path):
        if file.endswith(".ply"):
            ply_file = os.path.join(ply_path, file)
            # print(f"正在捕获视图: {ply_file}")
            if ply_file.endswith("_BC.ply"):
                capture_view(ply_file, camera_json_path, "BadCase.png")
            elif ply_file.endswith("_GT.ply"):
                capture_view(ply_file, camera_json_path, "GroundTruth.png")
            elif ply_file.endswith("_PS.ply"):
                capture_view(ply_file, camera_json_path, "Process.png")
                
                
    # def combine_images_with_text():
    # 合并3张png图片,在每张图上加上文字描述
    try:
        bad_case = Image.open("BadCase.png")      
        process = Image.open("Process.png")
        ground_truth = Image.open("GroundTruth.png")
        
        # 为文字添加额外的空间，增加高度以适应更大字体
        text_height = 100  # 增加文字区域高度
        font_size = 50  # 增大字体大小
        
        # 创建一个新的图像，宽度为3倍的单个图像宽度，高度为单个图像高度加上文字空间
        combined_width = bad_case.width + process.width + ground_truth.width
        combined_height = max(bad_case.height, process.height, ground_truth.height) + text_height
        combined_image = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(combined_image)
        
        # 尝试加载字体，使用更大字号
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # 大幅增大字体大小
        except IOError:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except:
                try:
                    # 尝试系统常见字体
                    system_fonts = ["DejaVuSans.ttf", "NotoSans-Regular.ttf", "FreeSans.ttf"]
                    for font_name in system_fonts:
                        try:
                            font = ImageFont.truetype(font_name, font_size)
                            break
                        except:
                            continue
                except:
                    font = ImageFont.load_default()
        
        # 在各个图像位置上方添加文字，居中对齐
        # 计算每张图片的中心位置
        x1_center = bad_case.width // 2
        x2_center = bad_case.width + process.width // 2
        x3_center = bad_case.width + process.width + ground_truth.width // 2
        
        # 文字内容
        text1 = "DGCNN (single-branch)"
        text2 = "DGCNN + HKS (dual-branch)"
        text3 = "Ground Truth"
        
        # 获取文本边界框来居中显示
        try:
            # 使用textbbox计算文本宽度（需要较新版本的PIL）
            text1_bbox = draw.textbbox((0, 0), text1, font=font)
            text2_bbox = draw.textbbox((0, 0), text2, font=font)
            text3_bbox = draw.textbbox((0, 0), text3, font=font)
            
            text1_width = text1_bbox[2] - text1_bbox[0]
            text2_width = text2_bbox[2] - text2_bbox[0]
            text3_width = text3_bbox[2] - text3_bbox[0]
        except AttributeError:
            # 兼容旧版PIL，估算文本宽度
            text1_width = len(text1) * font_size // 2
            text2_width = len(text2) * font_size // 2
            text3_width = len(text3) * font_size // 2
        
        # 居中绘制文字
        draw.text((x1_center - text1_width // 2, text_height // 4), text1, fill=(0, 0, 0), font=font)
        draw.text((x2_center - text2_width // 2, text_height // 4), text2, fill=(0, 0, 0), font=font)
        draw.text((x3_center - text3_width // 2, text_height // 4), text3, fill=(0, 0, 0), font=font)
      
        # 将三张图片粘贴到新图像中，位置在文字下方
        combined_image.paste(bad_case, (0, text_height))
        combined_image.paste(process, (bad_case.width, text_height))
        combined_image.paste(ground_truth, (bad_case.width + process.width, text_height))
        
        # 保存合并后的图像
        combined_image.save("CombinedView.png")
        print("已保存合并图像为 CombinedView.png")
    except Exception as e:
        print(f"合并图像时出错: {e}")

def main():
    if len(sys.argv) < 2:
        print("python visualize_Ablation_Experiment.py <mode> <ply_path> [camera_json_path]")
        print("e. python visualize_Ablation_Experiment.py v Ablation_Experiment1/X2J0T3NG_upper_colored_BK.ply ")
        print("e. python visualize_Ablation_Experiment.py c Ablation_Experiment1 ")
        print("e. python visualize_Ablation_Experiment.py c Ablation_Experiment1 camera_view.json")
        return
    mode = sys.argv[1] if len(sys.argv) > 1 else "v"
    print(f"当前模式: {mode}")
    
    if mode == "v": # View mode
        ply_path = sys.argv[2]
        run_view_mode(ply_path)
        
    elif mode == "c":  # Capture mode
        ply_path = sys.argv[2]
        camera_json_path = sys.argv[3] if len(sys.argv) > 3 else "camera_view.json"
        run_capture_mode(ply_path, camera_json_path)
    else:
        print("无效模式，请选择 'v' 或 'c'")
        return
    
if __name__ == "__main__":
    main()