import open3d as o3d
import numpy as np
import json

def main():
    ply_path = "Ablation_Experiment1/X2J0T3NG_upper_colored_GT.ply"
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"无法读取三角网格文件: {ply_path}")
        return

    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    # 将颜色四舍五入到小数点后3位，避免浮点误差
    rounded_colors = np.round(colors, 3)
    # 找到所有唯一颜色
    unique_colors = np.unique(rounded_colors, axis=0)

    centers = []
    for color in unique_colors:
        mask = np.all(rounded_colors == color, axis=1)
        group_vertices = vertices[mask]
        if len(group_vertices) == 0:
            continue
        center = group_vertices.mean(axis=0)
        centers.append((color.tolist(), center.tolist()))
        # 准备保存到JSON文件
        with open("color_centers.json", "w") as f:
            json.dump(centers, f, indent=2)
        # print(f"{color.tolist()}中心点: {center.tolist()}")

    # 如需保存到文件，可取消下方注释
    # import json
    # with open("color_centers.json", "w") as f:
    #     json.dump(centers, f, indent=2)

if __name__ == "__main__":
    main()
