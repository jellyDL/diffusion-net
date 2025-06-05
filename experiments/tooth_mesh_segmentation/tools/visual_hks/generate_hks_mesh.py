import os
import numpy as np
import open3d as o3d
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.sparse.linalg._eigen.arpack


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
            eigenvalues, eigenvectors = slinalg.eigsh(L, k=num_eigenvalues, which='SM', 
                                                     maxiter=300000, tol=1e-5)
            break
        except arpack.ArpackNoConvergence as e:
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


def visualize_hks(mesh, hks_features, time_idx=0, output_dir=None, colormap_name='jet'):
    """可视化HKS特征，使用增强对比度的方式"""
    # 归一化HKS特征到[0,1]区间
    hks = hks_features[:, time_idx]
    
    # 使用更强的对比度增强 - 对牙齿模型使用更窄的百分位范围
    p_low, p_high = np.percentile(hks, [5, 95])  # 调整为5-95百分位，增强中间值的差异
    hks_clipped = np.clip(hks, p_low, p_high)
    hks_normalized = (hks_clipped - p_low) / (p_high - p_low)
    
    # 应用非线性映射增强对比度
    hks_enhanced = np.power(hks_normalized, 0.7)  # gamma校正，增强中等强度的对比
    
    # 使用对比度更鲜明的颜色映射
    colormap = get_cmap(colormap_name)
    colors = colormap(hks_enhanced)[:, :3]
    
    # 设置网格顶点颜色
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = mesh.vertices
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    o3d.visualization.draw_geometries([colored_mesh])
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"hks_t{time_idx}.ply")
        o3d.io.write_triangle_mesh(output_path, colored_mesh)
        print(f"Saved visualization to {output_path}")
    
    return colored_mesh


# 添加新函数，显示不同时间尺度之间的差异
def visualize_hks_differences(mesh, hks_features, time_steps, output_dir=None):
    """可视化不同时间尺度间的HKS差异"""
    
    # 选择几组有代表性的比较（例如最小vs中等，中等vs最大）
    comparisons = [
        (0, len(time_steps) // 2),  # 最小时间尺度 vs 中间时间尺度
        (len(time_steps) // 2, len(time_steps) - 1),  # 中间时间尺度 vs 最大时间尺度
        (0, len(time_steps) - 1)  # 最小时间尺度 vs 最大时间尺度
    ]
    
    for i, j in comparisons:
        hks_diff = np.abs(hks_features[:, i] - hks_features[:, j])
        
        # 归一化差异
        p_low, p_high = np.percentile(hks_diff, [2, 98])
        hks_diff_normalized = np.clip((hks_diff - p_low) / (p_high - p_low), 0, 1)
        
        # 使用"hot"颜色映射突显差异区域
        colormap = get_cmap('hot')
        colors = colormap(hks_diff_normalized)[:, :3]
        
        # 设置网格顶点颜色
        colored_mesh = o3d.geometry.TriangleMesh()
        colored_mesh.vertices = mesh.vertices
        colored_mesh.triangles = mesh.triangles
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # 可视化
        print(f"Visualizing difference between t={time_steps[i]} and t={time_steps[j]}")
        o3d.visualization.draw_geometries([colored_mesh])
        
        # 保存结果
        if output_dir:
            diff_path = os.path.join(output_dir, f"hks_diff_t{i}_t{j}.ply")
            o3d.io.write_triangle_mesh(diff_path, colored_mesh)
            print(f"Saved difference visualization to {diff_path}")


def visualize_tooth_features(mesh, hks_features, time_steps, output_dir=None):
    """专门为牙齿模型优化的HKS可视化函数，突显从局部到全局的特征变化"""
    
    # 为局部细节使用更小的时间尺度
    local_idx = 0  # 最小时间尺度
    hks = hks_features[:, local_idx]
    p_low, p_high = np.percentile(hks, [1, 99])
    hks_normalized = np.clip((hks - p_low) / (p_high - p_low), 0, 1)
    # 增强对比度，突显局部细节
    hks_enhanced = np.power(hks_normalized, 0.5)
    
    # 设置顶点颜色
    local_mesh = o3d.geometry.TriangleMesh()
    local_mesh.vertices = mesh.vertices
    local_mesh.triangles = mesh.triangles
    colormap = get_cmap('jet')
    local_mesh.vertex_colors = o3d.utility.Vector3dVector(colormap(hks_enhanced)[:, :3])
    
    # 创建可视化窗口并设置为白色背景(局部特征)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="牙齿HKS特征：局部细节", width=1280, height=720)
    vis.add_geometry(local_mesh)
    
    # 设置渲染选项 - 白色背景
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    render_option.light_on = True
    
    # 自适应窗口视图
    vis.get_view_control().set_zoom(0.8)
    
    # 保存并显示
    if output_dir:
        local_path = os.path.join(output_dir, "tooth_local_features.ply")
        o3d.io.write_triangle_mesh(local_path, local_mesh)
        
    vis.run()
    vis.destroy_window()
    
    # 为全局结构使用更大的时间尺度
    global_idx = len(time_steps) - 1  # 最大时间尺度
    hks = hks_features[:, global_idx]
    p_low, p_high = np.percentile(hks, [5, 95])
    hks_normalized = np.clip((hks - p_low) / (p_high - p_low), 0, 1)
    
    # 设置顶点颜色
    global_mesh = o3d.geometry.TriangleMesh()
    global_mesh.vertices = mesh.vertices
    global_mesh.triangles = mesh.triangles
    colormap = get_cmap('plasma')
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(colormap(hks_normalized)[:, :3])
    
    # 创建可视化窗口并设置为白色背景(全局特征)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="牙齿HKS特征：全局结构", width=1280, height=720)
    vis.add_geometry(global_mesh)
    
    # 设置渲染选项 - 白色背景
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    render_option.light_on = True
    
    # 自适应窗口视图
    vis.get_view_control().set_zoom(0.8)
    
    # 保存并显示
    if output_dir:
        global_path = os.path.join(output_dir, "tooth_global_features.ply")
        o3d.io.write_triangle_mesh(global_path, global_mesh)
    
    vis.run()
    vis.destroy_window()
    
    return local_mesh, global_mesh


def visualize_hks_consistent(mesh, hks_features, time_steps, output_dir=None):
    """使用全局一致的归一化方法可视化不同时间尺度的HKS特征，使结果更易于比较"""
    
    # 首先找出所有时间尺度下HKS值的全局最小值和最大值
    all_hks = hks_features.flatten()
    # 使用较为温和的百分位裁剪，避免极端值影响
    global_min, global_max = np.percentile(all_hks, [10, 90])
    
    # 使用一致的颜色映射
    colormap_name = 'viridis'
    colormap = get_cmap(colormap_name)
    
    # 依次可视化每个时间尺度
    for t_idx, t in enumerate(time_steps):
        print(f"Consistently visualizing time step t={t}")
        hks = hks_features[:, t_idx]
        
        # 使用全局统一的归一化
        hks_normalized = np.clip((hks - global_min) / (global_max - global_min), 0, 1)
        
        # 轻微的非线性调整，以平衡不同尺度的视觉效果
        gamma = 0.9  # 更接近线性映射
        hks_enhanced = np.power(hks_normalized, gamma)
        
        # 设置网格顶点颜色
        colored_mesh = o3d.geometry.TriangleMesh()
        colored_mesh.vertices = mesh.vertices
        colored_mesh.triangles = mesh.triangles
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colormap(hks_enhanced)[:, :3])
        
        # 可视化
        o3d.visualization.draw_geometries([colored_mesh])
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"hks_consistent_t{t_idx}.ply")
            o3d.io.write_triangle_mesh(output_path, colored_mesh)
            print(f"Saved consistent visualization to {output_path}")
    
    # 返回不同时间尺度下的最小和最大值，以便查看数值范围差异
    time_step_stats = []
    for t_idx in range(len(time_steps)):
        hks = hks_features[:, t_idx]
        p_min, p_max = np.percentile(hks, [10, 90])
        time_step_stats.append((p_min, p_max))
    
    print("\nHKS值范围统计（10-90百分位）:")
    for t_idx, (min_val, max_val) in enumerate(time_step_stats):
        print(f"  t={time_steps[t_idx]}: {min_val:.6f} - {max_val:.6f}, 范围: {max_val-min_val:.6f}")
    
    return time_step_stats


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


def explain_hks_color_meaning():
    """详细解释HKS热力图中不同颜色深浅的具体含义"""
    print("\n========== HKS热力图颜色含义详解 ==========")
    print("在HKS热力图中，颜色代表了热在曲面上扩散的行为特征:")
    
    print("\n1. 颜色映射基本含义:")
    print("   • 红/黄色区域 (高值): 热量保留较长时间的区域")
    print("   • 蓝/青色区域 (低值): 热量扩散较快的区域")
    
    print("\n2. 几何特征与颜色的关系:")
    print("   • 尖锐凸起: 通常呈现为红/黄色热点，因为热量从尖端扩散较慢")
    print("   • 平坦区域: 通常呈现为青色或浅色区域，热量均匀扩散")
    print("   • 凹陷区域: 通常呈现为深蓝色，热量容易聚集并快速扩散")
    print("   • 边缘和轮廓: 在中等时间尺度下呈现为较明显的颜色对比")
    
    print("\n3. 牙齿模型特有颜色模式:")
    print("   • 咬合面纹理: 小尺度下显示为红黄色细小图案，代表微小凸起和沟壑")
    print("   • 牙冠边缘: 中等尺度下呈现为明显的颜色边界")
    print("   • 牙根区域: 大尺度下通常颜色较为均匀，与牙冠形成对比")
    print("   • 尖牙尖端: 在所有尺度下通常保持较高值(红/黄色)")
    
    print("\n4. 不同时间尺度下的颜色变化:")
    print("   • 小时间尺度 (t≤0.1): 颜色变化剧烈，微小几何细节都能引起颜色差异")
    print("   • 中等时间尺度 (0.1<t≤10): 主要结构特征形成稳定的颜色模式")
    print("   • 大时间尺度 (t>10): 颜色变化平缓，只有主要结构差异才能看到颜色差别")
    
    print("\n5. 颜色变化率的意义:")
    print("   • 颜色急剧变化区域: 表示几何结构突变处，如边缘或拐角")
    print("   • 颜色渐变区域: 表示几何特征平滑过渡")
    print("   • 颜色一致区域: 表示几何特征相似的连续区域")
    
    print("\n6. 特征差异图特有颜色含义:")
    print("   • 亮红/黄色区域: 局部特征与全局特征差异最大的区域")
    print("   • 暗色/蓝色区域: 在不同时间尺度下表现相似的区域")
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
        # 解释热力图中颜色的具体含义
        explain_hks_color_meaning()
        
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
            vis.run()
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
        
        # 创建特征差异可视化
        def visualize_feature_differences(mesh, hks_features, time_steps, output_dir=None):
            """可视化不同时间尺度HKS特征的相对差异"""
            # 选择最小和最大时间尺度计算相对变化
            min_idx, max_idx = 0, len(time_steps) - 1
            
            # 计算归一化的相对变化率
            min_hks = hks_features[:, min_idx]
            max_hks = hks_features[:, max_idx]
            
            # 避免除零错误
            epsilon = np.mean(min_hks) * 0.001
            relative_change = np.abs(max_hks - min_hks) / (min_hks + epsilon)
            
            # 归一化并裁剪极端值
            p_low, p_high = np.percentile(relative_change, [5, 95])
            relative_change_norm = np.clip((relative_change - p_low) / (p_high - p_low), 0, 1)
            
            # 使用热图颜色映射显示变化大的区域
            colormap = get_cmap('hot')
            colors = colormap(relative_change_norm)[:, :3]
            
            # 设置网格顶点颜色
            colored_mesh = o3d.geometry.TriangleMesh()
            colored_mesh.vertices = mesh.vertices
            colored_mesh.triangles = mesh.triangles
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # 创建可视化窗口并设置为白色背景
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"从局部到全局的特征变化(t={time_steps[min_idx]}→t={time_steps[max_idx]})", width=1280, height=720)
            vis.add_geometry(colored_mesh)
            
            # 设置渲染选项 - 白色背景
            render_option = vis.get_render_option()
            render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
            render_option.light_on = True
            
            # 自适应窗口视图
            view_control = vis.get_view_control()
            view_control.set_zoom(0.8)
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
            # 保存结果
            if output_dir:
                output_path = os.path.join(output_dir, "hks_feature_change.ply")
                o3d.io.write_triangle_mesh(output_path, colored_mesh)
                print(f"Saved feature difference visualization to {output_path}")
            
            return colored_mesh
        
        # 执行特征差异可视化
        visualize_feature_differences(mesh, hks_features, time_steps, output_dir)
        
        # 如果需要查看不同颜色映射的效果，可以取消以下注释
        '''
        # 高级可视化：使用不同颜色映射可以提供额外的视觉角度
        print("\n可选：使用多种颜色映射进行可视化...")
        colormap_names = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        for colormap_name in colormap_names:
            print(f"\n使用 {colormap_name} 颜色映射进行全局特征可视化...")
            visualize_hks(mesh, hks_features, -1, output_dir, colormap_name=colormap_name)
        '''
        
        # 显示不同时间尺度间的差异
        print("Visualizing differences between time scales...")
        # visualize_hks_differences(mesh, hks_features, time_steps, output_dir)
        
        # 专门为牙齿模型优化的可视化
        print("Visualizing tooth-specific features...")
        visualize_tooth_features(mesh, hks_features, time_steps, output_dir)
        
        # 添加一致化的可视化，使不同时间尺度可比较
        print("\n使用全局一致的归一化方法可视化HKS特征...")
        # visualize_hks_consistent(mesh, hks_features, time_steps, output_dir)
        
        return hks_features
    except Exception as e:
        print(f"计算HKS特征时出错: {e}")
        print("尝试更高的网格简化比例或检查网格质量")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_hks.py <obj_file_path> [output_directory] [simplify_ratio]")
        sys.exit(1)
    
    obj_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    simplify_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    main(obj_file, output_dir, simplify=True, target_ratio=simplify_ratio)
