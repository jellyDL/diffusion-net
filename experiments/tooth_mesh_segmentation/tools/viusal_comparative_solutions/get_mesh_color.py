import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_vertex_colors_from_ply(ply_file_path):
    """
    从PLY文件中获取所有顶点的颜色值
    
    Args:
        ply_file_path (str): PLY文件路径
    
    Returns:
        dict: 包含顶点颜色信息的字典
    """
    # 检查文件是否存在
    if not os.path.exists(ply_file_path):
        print(f"错误: 文件 '{ply_file_path}' 不存在")
        return None
    
    try:
        # 加载PLY文件
        print(f"正在加载PLY文件: {ply_file_path}")
        mesh = o3d.io.read_triangle_mesh(ply_file_path)
        
        # 检查网格信息
        print(f"网格顶点数量: {len(mesh.vertices)}")
        print(f"网格面数量: {len(mesh.triangles)}")
        print(f"网格是否有顶点颜色: {mesh.has_vertex_colors()}")
        
        result = {
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.triangles),
            "has_vertex_colors": mesh.has_vertex_colors(),
            "colors": None,
            "unique_colors": None,
            "color_statistics": None
        }
        
        if mesh.has_vertex_colors():
            # 获取顶点颜色
            colors = np.asarray(mesh.vertex_colors)
            result["colors"] = colors
            
            print(f"顶点颜色数组形状: {colors.shape}")
            print(f"颜色值范围: R({colors[:, 0].min():.3f}-{colors[:, 0].max():.3f}), "
                  f"G({colors[:, 1].min():.3f}-{colors[:, 1].max():.3f}), "
                  f"B({colors[:, 2].min():.3f}-{colors[:, 2].max():.3f})")
            
            # 显示前10个顶点的颜色值
            print("\n前10个顶点的颜色值 (RGB):")
            for i in range(min(10, len(colors))):
                r, g, b = colors[i]
                print(f"  顶点 {i}: R={r:.3f}, G={g:.3f}, B={b:.3f}")
            
            # 找到所有唯一的颜色值
            unique_colors = np.unique(colors.reshape(-1, 3), axis=0)
            result["unique_colors"] = unique_colors
            
            print(f"\n唯一颜色数量: {len(unique_colors)}")
            print("所有唯一颜色值:")
            for i, color in enumerate(unique_colors):
                count = np.sum(np.all(colors == color, axis=1))
                # print(f"  颜色 {i+1}: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}) - 出现次数: {count}")
                print(f"\"{i+1}\":[{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}],")
            
            # 统计信息
            result["color_statistics"] = {
                "unique_color_count": len(unique_colors),
                "color_counts": []
            }
            
            for color in unique_colors:
                count = np.sum(np.all(colors == color, axis=1))
                result["color_statistics"]["color_counts"].append({
                    "rgb": color.tolist(),
                    "count": count
                })
        else:
            print("网格没有顶点颜色信息")
        
        return result
        
    except Exception as e:
        print(f"处理PLY文件时出错: {e}")
        return None

def visualize_colors(color_info, ply_file):
    """
    可视化颜色信息，生成颜色图表
    
    Args:
        color_info (dict): 颜色信息字典
        ply_file (str): PLY文件名
    """
    if not color_info['has_vertex_colors']:
        print("没有颜色信息，无法生成可视化图表")
        return
    
    stats = color_info['color_statistics']
    color_counts = stats['color_counts']
    
    # 按出现次数排序
    color_counts_sorted = sorted(color_counts, key=lambda x: x['count'], reverse=True)
    
    # 创建图表 - 宽度缩为原来的30%
    fig, ax = plt.subplots(1, 1, figsize=(3, 8))
    
    # 颜色色块展示
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(color_counts_sorted))
    ax.set_title(f'颜色色块展示 - {os.path.basename(ply_file)}')
    ax.set_xlabel('RGB值')
    ax.set_ylabel('颜色序号')
    
    for i, item in enumerate(color_counts_sorted):
        rgb = item['rgb']
        count = item['count']
        
        # 绘制色块 - 调整位置和大小适应新的宽度
        rect = patches.Rectangle((0, len(color_counts_sorted)-i-1), 1.0, 0.6, 
                               linewidth=1, edgecolor='black', facecolor=rgb)
        ax.add_patch(rect)
        
        # 添加RGB值和数量标签 - 调整位置向右移动
        ax.text(1.1, len(color_counts_sorted)-i-0.3, 
                f'RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})', 
                va='center', fontsize=6, fontweight='bold')
        
        # 修复格式化字符串错误，先计算值再格式化
        rgb_255 = [val * 255.0 for val in rgb]
        ax.text(1.1, len(color_counts_sorted)-i-0.5, 
                f'RGB255({rgb_255[0]:.0f}, {rgb_255[1]:.0f}, {rgb_255[2]:.0f})', 
                va='center', fontsize=6)
        
        ax.text(1.1, len(color_counts_sorted)-i-0.7, f'数量: {count}', 
                va='center', fontsize=6)
    
    ax.set_xlim(0, 5)
    ax.set_yticks(range(len(color_counts_sorted)))
    # 设置y轴标签并上移
    ax.set_yticks([i+0.35 for i in range(len(color_counts_sorted))])  # 上移标签位置
    ax.set_yticklabels([f'颜色{i+1}' for i in range(len(color_counts_sorted))])
    
    # 移除tight_layout，直接调整布局
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    
    # # 保存图表
    # output_image = f"{os.path.splitext(ply_file)[0]}_color_visualization.png"
    # plt.savefig(output_image, dpi=300, bbox_inches='tight')
    # print(f"颜色可视化图表已保存到: {output_image}")
    
    plt.show()

def main():

    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
    else:
        print("exp. python get_mesh_color.py <path_to_ply_file>")
        return 
    
    # 获取顶点颜色信息
    color_info = get_vertex_colors_from_ply(ply_file)
    
    if color_info:
        print(f"\n=== 颜色信息汇总 ===")
        print(f"顶点总数: {color_info['vertex_count']}")
        print(f"是否有颜色: {color_info['has_vertex_colors']}")
        
        if color_info['has_vertex_colors']:
            stats = color_info['color_statistics']
            print(f"唯一颜色数: {stats['unique_color_count']}")
            
            # # 保存颜色信息到文件
            # output_file = f"{os.path.splitext(ply_file)[0]}_color_info.txt"
            # with open(output_file, 'w') as f:
            #     f.write(f"PLY文件: {ply_file}\n")
            #     f.write(f"顶点总数: {color_info['vertex_count']}\n")
            #     f.write(f"面片总数: {color_info['triangle_count']}\n")
            #     f.write(f"唯一颜色数: {stats['unique_color_count']}\n\n")
                
            #     f.write("所有唯一颜色及其出现次数:\n")
            #     for i, color_stat in enumerate(stats['color_counts']):
            #         rgb = color_stat['rgb']
            #         count = color_stat['count']
            #         f.write(f"颜色 {i+1}: RGB({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}) - 出现次数: {count}\n")
            
            # print(f"\n颜色信息已保存到: {output_file}")
            
            # 生成颜色可视化图表
            visualize_colors(color_info, ply_file)

if __name__ == "__main__":
    main()

