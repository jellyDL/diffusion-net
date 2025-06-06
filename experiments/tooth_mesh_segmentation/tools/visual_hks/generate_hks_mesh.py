import os
import sys
import time
import numpy as np
import open3d as o3d
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence


def load_mesh(obj_path, simplify=True, target_ratio=0.2):
    """加载OBJ文件并返回网格，可选择进行网格简化
    
    Args:
        obj_path: OBJ文件路径
        simplify: 是否进行网格简化
        target_ratio: 简化后的网格三角形数量与原始网格的比例
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    original_triangle_count = len(mesh.triangles)
    
    if simplify and original_triangle_count > 0:
        print(f"Original mesh: {original_triangle_count} triangles")
        target_triangles = int(original_triangle_count * target_ratio)
        print(f"Simplifying mesh to {target_triangles} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_triangles)
        print(f"Simplified mesh: {len(mesh.triangles)} triangles")
    
    mesh.compute_vertex_normals()
    return mesh


def compute_laplacian(mesh):
    """计算拉普拉斯-贝尔特拉米算子（使用Cotangent权重）"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    n_vertices = len(vertices)
    
    # 构建邻接矩阵
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], 
                        faces[:, 1], faces[:, 2], faces[:, 0], 
                        faces[:, 2], faces[:, 0], faces[:, 1]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], 
                        faces[:, 0], faces[:, 1], faces[:, 2], 
                        faces[:, 1], faces[:, 2], faces[:, 0]])
    
    # 计算边长和角度
    def compute_edge_length(v1, v2):
        return np.linalg.norm(vertices[v1] - vertices[v2], axis=1)
    
    edge_01 = compute_edge_length(faces[:, 0], faces[:, 1])
    edge_12 = compute_edge_length(faces[:, 1], faces[:, 2])
    edge_20 = compute_edge_length(faces[:, 2], faces[:, 0])
    
    # 计算余切权重（Cotangent weights）
    def compute_cot(a, b, c):
        # 计算余切值
        ab = vertices[a] - vertices[b]
        ac = vertices[a] - vertices[c]
        numerator = np.sum(ab * ac, axis=1)
        denominator = np.linalg.norm(np.cross(ab, ac), axis=1)
        return numerator / np.maximum(denominator, 1e-10)
    
    cot_0 = compute_cot(faces[:, 0], faces[:, 1], faces[:, 2])
    cot_1 = compute_cot(faces[:, 1], faces[:, 2], faces[:, 0])
    cot_2 = compute_cot(faces[:, 2], faces[:, 0], faces[:, 1])
    
    # 权重
    weights = np.concatenate([cot_2, cot_0, cot_1, cot_2, cot_0, cot_1, cot_2, cot_0, cot_1]) / 2.0
    
    # 构建稀疏矩阵
    W = sparse.coo_matrix((weights, (i, j)), shape=(n_vertices, n_vertices))
    W = W.tocsr()
    
    # 计算质量矩阵（Mass matrix）
    areas = np.zeros(n_vertices)
    for f in faces:
        v0, v1, v2 = vertices[f]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas[f] += area / 3.0
    
    M = sparse.diags(areas)
    
    # 构建拉普拉斯矩阵
    L = sparse.diags(W.sum(axis=1).flat) - W
    
    # 归一化的拉普拉斯矩阵
    Minv = sparse.diags(1.0 / areas)
    L_norm = Minv @ L
    
    return L_norm, M


def compute_hks(L, num_eigenvalues=100, time_steps=None):
    """计算HKS特征"""
    if time_steps is None:
        time_steps = [0.1, 1, 10, 100]
    
    # 降低默认特征值数量
    original_num_ev = num_eigenvalues
    
    # 尝试计算特征值和特征向量，如果失败则降低特征值数量
    while num_eigenvalues >= 10:
        try:
            print(f"尝试计算 {num_eigenvalues} 个特征值...")
            # 增加最大迭代次数并设置更高的收敛容差
            start_time = time.time()
            eigenvalues, eigenvectors = slinalg.eigsh(
                L, k=num_eigenvalues, which='SM',
                maxiter=30000, tol=1e-3
            )
            elapsed = time.time() - start_time
            print(f"特征分解耗时: {elapsed:.2f} 秒")
            break
        except ArpackNoConvergence as e:
            # 使用已收敛的部分特征值和特征向量
            print(f"部分收敛: {len(e.eigenvalues)}/{num_eigenvalues} 个特征值已收敛")
            if len(e.eigenvalues) >= 10:
                eigenvalues = e.eigenvalues
                eigenvectors = e.eigenvectors
                print(f"使用已收敛的 {len(eigenvalues)} 个特征值继续计算...")
                break
            else:
                print(f"收敛的特征值太少，尝试降低请求数量")
                num_eigenvalues = num_eigenvalues // 2
                print(f"降低到 {num_eigenvalues} 个特征值...")
        except Exception as e:
            print(f"计算 {num_eigenvalues} 个特征值失败: {str(e)}")
            num_eigenvalues = num_eigenvalues // 2
            print(f"降低到 {num_eigenvalues} 个特征值...")
    
    if num_eigenvalues < 10 and 'eigenvalues' not in locals():
        raise ValueError("无法计算足够的特征值，请尝试进一步简化网格。")
    
    # 确保特征值是正的
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 计算HKS
    hks_features = np.zeros((eigenvectors.shape[0], len(time_steps)))
    
    for i, t in enumerate(time_steps):
        # HKS在时间t的公式： sum_j exp(-λ_j * t) * φ_j(x)^2
        weights = np.exp(-eigenvalues * t)
        hks = np.sum(eigenvectors[:, :] ** 2 * weights, axis=1)
        hks_features[:, i] = hks
    
    return hks_features


def explain_hks_heatmap(time_steps):
    """解释HKS热力图所表达的几何意义"""
    print("\n========== HKS热力图解释 ==========")
    print("HKS(Heat Kernel Signature)热力图的含义:")
    
    print("\n1. 基本原理:")
    print("   - HKS模拟热从一点扩散到整个曲面的过程")
    print("   - 不同的时间参数t对应热扩散的不同阶段")
    print("   - 热量在几何特征处的扩散行为不同，因此能够捕捉到模型的结构特征")
    
    print("\n2. 颜色映射含义:")
    print("   - 热点区域(红色/黄色)：热量聚集处，通常对应几何特征如凸起、边缘")
    print("   - 冷点区域(蓝色/绿色)：热量扩散较快处，通常对应平坦或凹陷区域")
    
    print("\n3. 不同时间尺度的特征解释:")
    for i, t in enumerate(time_steps):
        if t <= 0.1:
            print(f"   - t={t}: 极小局部细节 - 显示牙齿表面的微小纹理、裂缝和尖锐特征")
        elif t <= 1.0:
            print(f"   - t={t}: 局部特征 - 显示牙齿咬合面的沟壑、釉质边缘等小尺度特征")
        elif t <= 10.0:
            print(f"   - t={t}: 中等尺度特征 - 显示整个牙冠、牙根等中等大小的解剖结构")
        elif t <= 100.0:
            print(f"   - t={t}: 大尺度特征 - 显示牙齿整体形状轮廓和主要解剖区域")
        else:
            print(f"   - t={t}: 全局特征 - 反映牙齿整体形状和相对位置特征")
    
    print("\n4. 牙齿模型特有的热力图特点:")
    print("   - 咬合面：通常在小时间尺度下会显示高强度值(热点)，因为有复杂纹理")
    print("   - 釉质边缘：在中等时间尺度下显示为边界特征")
    print("   - 牙根：在大时间尺度下与牙冠区分更明显")
    print("   - 平滑表面：在所有时间尺度下通常表现为均匀的低强度区域")
    
    print("\n5. HKS特性:")
    print("   - 变形不变性：同一牙齿在不同姿态下，HKS基本保持一致")
    print("   - 多尺度性：从微观特征到宏观结构都能捕捉")
    print("   - 局部性：小时间参数主要受局部几何影响，不受远处几何变化影响")
    
    print("\n6. 实际应用:")
    print("   - 牙齿分类与识别：不同类型牙齿(门牙、犬齿、磨牙)有不同HKS特征")
    print("   - 结构分析：可用于评估牙齿解剖特征")
    print("   - 对比分析：可比较不同牙齿模型或同一牙齿的不同扫描结果")
    print("================================================\n")


def explain_time_step_effect(t_val):
    """为特定的time_step值提供可视化效果的详细说明"""
    print(f"\n========== time_step t={t_val} 的可视化效果说明 ==========")
    
    if t_val <= 0.01:
        print("【极小局部特征 (t≤0.01)】")
        print("- 当前可视化展示的是极小尺度的几何特征，如微小的表面纹理和细节")
        print("- 红/黄色区域：微小的凸起、锐利边缘和精细纹理")
        print("- 蓝色区域：平滑过渡区域和微小凹陷")
        print("- 在牙齿模型上，您应该能看到：")
        print("  • 咬合面上的微小沟壑和裂缝显示为红/黄色")
        print("  • 釉质表面微小的不规则性")
        print("  • 齿尖边缘的精细轮廓")
        print("- 这个尺度主要捕捉牙齿表面的微小变化，对应人类视觉中的'细节观察'")
    
    elif t_val <= 1.0:
        print("【局部特征 (0.01<t≤1.0)】")
        print("- 当前可视化展示的是局部区域的几何特征，如主要表面起伏和小型解剖结构")
        print("- 红/黄色区域：局部凸起、齿尖、咬合面边缘")
        print("- 蓝色区域：较平缓的表面区域和小型凹陷")
        print("- 在牙齿模型上，您应该能看到：")
        print("  • 咬合面轮廓和主要沟壑模式")
        print("  • 牙齿主要边缘处的颜色变化")
        print("  • 釉质与牙龈交界处的颜色对比")
        print("- 这个尺度能够捕捉牙齿的基本解剖特征，对应人类视觉中的'近距离观察'")
    
    elif t_val <= 100.0:
        print("【中等尺度特征 (1.0<t≤100.0)】")
        print("- 当前可视化展示的是整体结构的几何特征，如牙齿的主要组成部分")
        print("- 红/黄色区域：牙冠、齿尖等主要凸出结构")
        print("- 蓝色区域：牙根、牙颈部等相对平缓区域")
        print("- 在牙齿模型上，您应该能看到：")
        print("  • 牙冠整体形状的突出显示")
        print("  • 牙齿不同解剖部位之间的明显区分")
        print("  • 咬合面整体高程与侧面的区别")
        print("- 这个尺度能够捕捉牙齿的整体组成特征，对应人类视觉中的'常规观察距离'")
    
    else:
        print("【全局特征 (t>100.0)】")
        print("- 当前可视化展示的是全局结构的几何特征，反映整个牙齿的形态")
        print("- 红/黄色区域：整体凸起和主导性结构")
        print("- 蓝色区域：整体凹陷和次要结构")
        print("- 在牙齿模型上，您应该能看到：")
        print("  • 牙齿整体轮廓和主要形状特征")
        print("  • 牙齿类型特有的宏观特征(如磨牙的宽大咬合面、门牙的扁平形态)")
        print("  • 脱离细节的整体形状表示")
        print("- 这个尺度反映了牙齿的整体几何特性，对应人类视觉中的'远距离观察'或'整体感知'")
    
    print("观察技巧：")
    print("- 注意颜色梯度变化，突变区域通常对应几何特征的边界")
    print("- 比较相同区域在不同time_step下的颜色变化，理解多尺度几何特征")
    print("- 红/黄色区域是该时间尺度下的'重要'几何特征")
    print("=================================================\n")


def main(obj_path, output_dir=None, simplify=True, target_ratio=0.2):
    """主函数"""
    # 加载网格
    print("Loading mesh...")
    mesh = load_mesh(obj_path, simplify=simplify, target_ratio=target_ratio)
    
    # 计算拉普拉斯矩阵
    print("Computing Laplacian...")
    L, M = compute_laplacian(mesh)
    
    # 计算HKS特征
    print("Computing HKS features...")
    # 使用更广泛且更精细的时间尺度范围
    time_steps = [0.001, 0.1, 10.0, 1000.0]  # 跨度更大的对数尺度
    try:
        # 增加特征值数量以捕获更多细节
        hks_features = compute_hks(L, num_eigenvalues=128, time_steps=time_steps)
        
        # 解释HKS热力图含义
        explain_hks_heatmap(time_steps)
        
        # 可视化不同时间尺度的HKS
        print("Visualizing HKS features with enhanced contrast...")
        
        # 先计算所有尺度下的HKS平均值和标准差，用于特征归一化
        all_features = []
        for t_idx in range(len(time_steps)):
            all_features.append(hks_features[:, t_idx])
        
        # 创建分层增强的可视化函数
        def visualize_with_enhanced_contrast(mesh, hks, t_idx, t_val, output_dir):
            # 根据时间尺度调整对比度增强参数
            if t_val <= 0.01:  # 极小局部特征
                p_low, p_high = np.percentile(hks, [2, 90])
                gamma = 0.5  # 增强小值区域对比度
            elif t_val <= 1.0:  # 局部特征
                p_low, p_high = np.percentile(hks, [5, 95])
                gamma = 0.6
            elif t_val <= 100.0:  # 中等尺度特征
                p_low, p_high = np.percentile(hks, [10, 98])
                gamma = 0.7
            else:  # 全局特征
                p_low, p_high = np.percentile(hks, [15, 99])
                gamma = 0.8
            
            # 裁剪和归一化
            hks_clipped = np.clip(hks, p_low, p_high)
            hks_normalized = (hks_clipped - p_low) / (p_high - p_low)
            
            # 分层增强 - 使用多次非线性变换增强可见性
            hks_enhanced = np.power(hks_normalized, gamma)
            
            # 应用颜色映射
            colormap = get_cmap('jet')
            colors = colormap(hks_enhanced)[:, :3]
            
            # 设置网格顶点颜色
            colored_mesh = o3d.geometry.TriangleMesh()
            colored_mesh.vertices = mesh.vertices
            colored_mesh.triangles = mesh.triangles
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # 创建可视化窗口并设置为白色背景
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"HKS at t={t_val} (优化对比度)", width=1280, height=720, visible=True)
            vis.add_geometry(colored_mesh)
            
            # 设置渲染选项 - 白色背景
            render_option = vis.get_render_option()
            render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
            render_option.light_on = True
            
            # 自适应窗口视图
            view_control = vis.get_view_control()
            view_control.set_zoom(0.8)
            
            # 运行可视化
            # vis.run()
            vis.destroy_window()
            
            # 保存结果
            if output_dir:
                output_path = os.path.join(output_dir, f"hks_enhanced_t{t_idx}.ply")
                o3d.io.write_triangle_mesh(output_path, colored_mesh)
                print(f"Saved enhanced visualization to {output_path}")
            
            return colored_mesh
        
        # 使用增强对比度方法可视化每个时间尺度
        for t_idx, t in enumerate(time_steps):
            # 提供当前time_step的可视化效果说明
            explain_time_step_effect(t)
            
            print(f"Visualizing time step t={t} with optimized contrast")
            visualize_with_enhanced_contrast(mesh, hks_features[:, t_idx], t_idx, t, output_dir)
        
        # 创建HKS特征对比分析图
        print("\n创建不同时间尺度HKS特征的对比图...")
        print("对比图说明: 此图展示了从最小时间尺度(局部细节)到最大时间尺度(全局特征)的变化")
        print("- 亮色区域: 随着时间尺度变化，几何特征表现差异最大的区域")
        print("- 暗色区域: 在不同时间尺度下保持相似特性的区域")
        print("- 这种可视化有助于理解哪些区域的几何特征是尺度相关的，哪些是尺度不变的")
        
        return hks_features
    except Exception as e:
        print(f"计算HKS特征时出错: {e}")
        print("尝试更高的网格简化比例或检查网格质量")
        return None


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_hks.py <obj_file_path> [output_directory] [simplify_ratio]")
        sys.exit(1)
    
    obj_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    simplify_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    main(obj_file, output_dir, simplify=True, target_ratio=simplify_ratio)
