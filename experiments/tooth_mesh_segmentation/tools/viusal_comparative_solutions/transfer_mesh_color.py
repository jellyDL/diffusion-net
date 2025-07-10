import open3d as o3d
import os
import numpy as np
import json
import sys

def get_mesh_colors_dict_from_json(json_file):
    with open(json_file, 'r') as f:
        try:
            color_dict = json.load(f)
            return color_dict
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析JSON文件 {json_file} - {e}")
            return []
        
def get_fdi_colors_dict_from_json(json_file):
    with open(json_file, 'r') as f:
        try:
            color_json = json.load(f)
            color_list = color_json['fdi_colors']
            # print("color_dict: ", type(color_list))
            # 将color_list 写入 color_dict字典
            color_dict = {}
            for item in color_list:
                toothnum = item['label']
                color = item['color']
                # 将颜色转换为浮点数列表
                color = [c / 255.0 for c in color]
                color_dict[toothnum] = [round(c, 3) for c in color]
            return color_dict
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析JSON文件 {json_file} - {e}")
            return []
    

def visualize_ply(ply_file_path):
    """
    使用Open3D可视化PLY文件
    
    参数:
        ply_file_path: PLY文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(ply_file_path):
        print(f"错误: 文件 {ply_file_path} 不存在")
        return
    
    # 加载PLY文件
    print(f"正在加载PLY文件: {ply_file_path}")
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    
    # 计算顶点法线（如果不存在）
    if not mesh.has_vertex_normals():
        print("计算顶点法线...")
        mesh.compute_vertex_normals()
    
    # 检查是否有顶点颜色
    if not mesh.has_vertex_colors():
        print("警告: PLY文件没有顶点颜色信息")
    else:
        print("保留原始顶点颜色...")
    
    color_dict     = get_mesh_colors_dict_from_json("mesh_color.json")
    # new_color_dict = get_fdi_colors_dict_from_json("../visual_dgcnn_vs_hks/fdi_number.json")
    new_color_dict = get_mesh_colors_dict_from_json("new_mesh_color.json")
    
    # # 遍历顶点，获取顶点颜色
    vertex_colors = np.asarray(mesh.vertex_colors)
    for i, color in enumerate(vertex_colors):
        #如果color在color_dict.values里面，return对应的key值(取float的前3位小数，不到3为补全到3位)
        # print("color: ", color)
        r = round(color[0], 3)
        g = round(color[1], 3)
        b = round(color[2], 3)  
        color =[r, g, b]
        if color in color_dict.values():
            toothnum = list(color_dict.keys())[list(color_dict.values()).index(color)]
            # print("toothnum ", toothnum)
            new_color = new_color_dict[toothnum]
            # new_color赋值给网格
            mesh.vertex_colors[i] = o3d.utility.Vector3dVector([new_color])[0]

    # 创建坐标系可视化
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    # 翻转网格180度 (先围绕Y轴旋转180度)
    R_y = mesh.get_rotation_matrix_from_xyz([0, np.pi, 0])  # 绕Y轴旋转180度
    mesh.rotate(R_y, center=mesh.get_center())
    
    # 再顺时针旋转180度 (围绕Z轴旋转180度)
    R_z = mesh.get_rotation_matrix_from_xyz([0, 0, np.pi])  # 绕Z轴旋转180度
    mesh.rotate(R_z, center=mesh.get_center())
    
    # 显示网格
    print("正在显示网格...")
    o3d.visualization.draw_geometries([mesh, coordinate_frame], 
                                     window_name="牙冠可视化",
                                     width=1280, height=960)
    
    # 还原旋转以保存原始方向的网格 (按相反顺序应用相同的旋转来还原)
    mesh.rotate(R_z, center=mesh.get_center())  # 再次应用Z轴旋转以还原
    mesh.rotate(R_y, center=mesh.get_center())  # 再次应用Y轴旋转以还原
    
    # 保存网格
    new_mesh_path = ply_file_path[0:-4]+"_newcolor.ply"
    print(f"正在保存新网格到: {new_mesh_path}")
    o3d.io.write_triangle_mesh(new_mesh_path, mesh)
    

if __name__ == "__main__":
    # 指定PLY文件路径
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
    else:
        print("python transfer_mesh_color.py <path_to_ply_file>")
        sys.exit(1)        
    
    # 如果路径不是绝对路径，假设它在当前目录
    if not os.path.isabs(ply_file):
        ply_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ply_file)
    
    visualize_ply(ply_file)
    print("可视化完成。")
