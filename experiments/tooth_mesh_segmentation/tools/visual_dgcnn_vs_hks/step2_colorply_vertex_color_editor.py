import os
import sys
import numpy as np
import pyvista as pv
import vtk

class VertexColorEditor:
    """网格顶点颜色编辑器，支持通过鼠标修改顶点颜色"""
    
    def __init__(self, mesh_file):
        # 保存原始文件路径
        self.mesh_file = mesh_file
        
        # 加载网格
        self.mesh = pv.read(mesh_file)
        print(f"已加载网格: {os.path.basename(mesh_file)}")
        print(f"顶点数量: {self.mesh.n_points}")
        
        # 初始化颜色数组 (RGBA)
        # 如果网格已有颜色，则使用现有颜色；否则初始化为灰色
        if 'RGB' in self.mesh.point_data:
            rgb = self.mesh.point_data['RGB']
            self.colors = np.zeros((self.mesh.n_points, 4))
            self.colors[:, 0:3] = rgb / 255.0  # 归一化到 0-1
            self.colors[:, 3] = 1.0  # 不透明度设为 1.0
            print("已加载网格原始颜色")
        else:
            self.colors = np.ones((self.mesh.n_points, 4)) * 0.7  # 灰色
            self.colors[:, 3] = 1.0  # 不透明度设为 1.0
            print("未找到原始颜色，使用默认灰色")
        
        # 记录修改过的顶点
        self.modified_vertices = []
        
        # 当前选择的颜色 (RGBA)
        self.current_color = [1.0, 1.0, 0.0, 1.0]  # 默认黄色
        
        # 创建可视化窗口 - 使用默认尺寸而非直接全屏
        self.plotter = pv.Plotter()
        
        # 添加批量选择模式
        self.batch_mode = False
        self.selection_radius = 0.02  # 默认选择半径
        self.last_selected_point = None
        
        # 添加颜色历史记录
        self.color_history = []
        
        # 标记为未保存
        self._saved = False
        
        # 添加三角面片显示模式跟踪
        self.show_facets = False
    
    def setup(self):
        """设置交互式场景"""
        # 将颜色数组正确应用到网格上
        self.mesh.point_data["colors"] = self.colors
        
        # 添加网格到场景，确保使用正确的颜色属性名
        self.actor = self.plotter.add_mesh(self.mesh, scalars="colors", rgb=True)
        
        # 设置为全屏模式 (在创建窗口后设置)
        if hasattr(self.plotter, 'window_size'):
            try:
                # 获取屏幕分辨率
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                
                # 设置窗口大小为屏幕大小
                self.plotter.window_size = [screen_width, screen_height]
                print(f"设置窗口大小为: {screen_width}x{screen_height}")
            except Exception as e:
                print(f"设置全屏失败: {e}")
    
        # 设置鼠标点击事件处理 - 修复参数列表
        def on_pick(mesh, picker_or_id):
            print("接收到点拾取事件，参数类型:", type(picker_or_id), "值:", picker_or_id)
            
            point_id = None
            
            # 处理不同类型的picker_or_id
            try:
                if hasattr(picker_or_id, 'GetPointId'):
                    # 如果是vtkPointPicker对象，直接获取点ID
                    point_id = picker_or_id.GetPointId()
                    print(f"从vtkPointPicker获取到点ID: {point_id}")
                elif isinstance(picker_or_id, (list, tuple)):
                    # 如果传入的是坐标列表，找到最近的点
                    point = np.array(picker_or_id)
                    distances = np.linalg.norm(self.mesh.points - point, axis=1)
                    point_id = np.argmin(distances)
                elif isinstance(picker_or_id, (int, np.integer)):
                    # 如果直接是整数ID
                    point_id = picker_or_id
                else:
                    # 尝试转换为整数
                    try:
                        point_id = int(picker_or_id)
                    except:
                        print(f"无法从 {type(picker_or_id)} 转换为整数点ID")
                        return False
            except Exception as e:
                print(f"处理点拾取参数时出错: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 检查是否成功获取了点ID
            if point_id is None:
                print("未能获取有效的点ID")
                return False
                    
            # 继续处理，确保point_id是整数
            if 0 <= point_id < self.mesh.n_points:
                # 应用当前颜色(添加调试信息)
                print(f"应用颜色前: 顶点 {point_id} 颜色为 {self.colors[point_id]}")
                print(f"应用颜色: {self.current_color}")
                
                # 确保颜色数组正确格式化为RGBA
                if len(self.current_color) == 3:
                    # 如果颜色只有RGB分量，添加透明度
                    color_to_apply = self.current_color + [1.0]
                else:
                    color_to_apply = self.current_color
                    
                self.colors[point_id] = color_to_apply
                print(f"T应用颜色后: 顶点 {point_id} 颜色为 {self.colors[point_id]}")
                
                # # 更新网格颜色并调用强化刷新方法
                # self.mesh.point_data["colors"] = self.colors
                # self.refresh_render(immediate=True)  # 使用立即更新
                
                print("TT0 self.modified_vertices", self.modified_vertices)
                
                # 修复: 将修改的顶点记录到实例变量 self.modified_vertices
                # 记录修改过的顶点
                if point_id not in self.modified_vertices:
                    self.modified_vertices.append(point_id)
                
                print("TT1 self.modified_vertices", self.modified_vertices)
                
                # 记录最后选中的点
                self.last_selected_point = point_id
                
                # 显示选中的点
                self.show_selected_point(point_id)
                
                print(f"已修改顶点 {point_id} 的颜色 为 {self.colors[point_id]}，总共修改 {len(self.modified_vertices)} 个顶点")
                
                # 更新网格颜色并调用强化刷新方法
                self.mesh.point_data["colors"] = self.colors
                self.refresh_render(immediate=True)  # 使用立即更新
                
            else:
                print(f"顶点ID超出范围: {point_id}, 范围应该是0-{self.mesh.n_points-1}")
                
            return False  # 保持拾取功能开启
        
        # 使用两种方法启用点拾取，增加成功率
        # 方法1：标准点拾取
        self.plotter.enable_point_picking(callback=on_pick, show_message=True, use_picker='cell')
        
        # 方法2：直接的VTK事件处理（作为备份）
        def on_click(obj, event):
            try:
                click_pos = obj.GetEventPosition()
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.01)
                
                if picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer):
                    pick_pos = picker.GetPickPosition()
                    distances = np.linalg.norm(self.mesh.points - pick_pos, axis=1)
                    point_id = np.argmin(distances)
                    
                    # 保存操作以支持撤销
                    self.save_state_for_undo()
                    
                    # 检查是否为批量模式
                    if self.batch_mode:
                        self.color_points_in_radius(point_id)
                    else:
                        # 直接在这里修改颜色，避免回调层级过多
                        if isinstance(point_id, int) and 0 <= point_id < self.mesh.n_points:
                            # 应用当前颜色
                            self.colors[point_id] = self.current_color
                            
                            # 强制立即更新
                            self.refresh_render(immediate=True)
                            
                            # 记录修改过的顶点
                            if point_id not in self.modified_vertices:
                                self.modified_vertices.append(point_id)
                            
                            print(f"已修改顶点 {point_id} 的颜色")
                        
                    self.last_selected_point = point_id
                    
                    # 添加选中顶点的视觉指示
                    self.show_selected_point(point_id)
            except Exception as e:
                print(f"点击处理错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 修复: 使用 add_observer 而不是 AddObserver
        self.plotter.iren.add_observer("LeftButtonPressEvent", on_click)
        
        # 键盘事件处理 - 颜色选择和功能键
        def key_press(key):
            print(f"键盘事件触发: {key}")  # 更好的调试输出
            
            # 保存当前相机位置 (完整信息)
            camera_pos = self.plotter.camera_position
            focal_point = self.plotter.camera.GetFocalPoint()
            view_up = self.plotter.camera.GetViewUp()
            distance = self.plotter.camera.GetDistance()
            
            # 颜色选择 - 统一添加透明度分量
            if key == '1':
                self.current_color = [1.0, 0.0, 0.0, 1.0]  # 红色
                print("当前颜色: 红色")
            elif key == 'g':
                self.current_color = [0.0, 1.0, 0.0, 1.0]  # 绿色(修复:添加透明度)
                print("当前颜色: 绿色")
            elif key == 'b':
                self.current_color = [0.0, 0.0, 1.0, 1.0]  # 蓝色(修复:添加透明度)
                print("当前颜色: 蓝色")
            elif key == 'y':
                self.current_color = [1.0, 1.0, 0.0, 1.0]  # 黄色
                print("当前颜色: 黄色")
            elif key == 'c':
                self.current_color = [0.0, 1.0, 1.0, 1.0]  # 青色
                print("当前颜色: 青色")
            elif key == 'm':
                self.current_color = [1.0, 0.0, 1.0]  # 洋红色
                print("当前颜色: 洋红色")
            elif key == 'w':
                self.current_color = [1.0, 1.0, 1.0, 1.0]  # 白色(修复:添加透明度)
                print("当前颜色: 白色")
            elif key == 'k':
                self.current_color = [0.0, 0.0, 0.0, 1.0]  # 黑色
                print("当前颜色: 黑色")
            # 功能键
            elif key == 'x':
                self.clear_changes()
            elif key == 's':
                self.save_mesh()  # 使用增强的保存功能
            elif key == 'p':
                self.save_screenshot()  # 添加保存截图功能
            elif key == 'a':
                self.toggle_batch_mode()
            elif key == '+':
                self.increase_selection_radius()
            elif key == '-':
                self.decrease_selection_radius()
            elif key == 'z':
                self.undo_last_change()
            elif key == 'i':
                self.pick_color_from_point()
            elif key == 'f':
                self.toggle_facets()  # 切换三角面片显示
            elif key == 'z':  # 添加切换全屏/窗口模式的快捷键
                self.toggle_fullscreen()
            elif key == 'q':
                self.quit()
    
        # 禁用PyVista可能与'r'键冲突的默认快捷键
        if hasattr(self.plotter, 'reset_key_events'):
            self.plotter.reset_key_events()
        
        # 使用自定义方法注册键盘事件，避免使用内置快捷键
        for key_char in ['1', 'g', 'b', 'y', 'c', 'm', 'w', 'k', 'x', 's', 'a', 'z', 'i', 'f', 'z', 'q']:
            # 为每个按键创建封闭的范围
            def create_callback(k=key_char):
                return lambda: key_press(k)
            
            self.plotter.add_key_event(key_char, create_callback())
        
        # 新增快捷键
        self.plotter.add_key_event('p', lambda: key_press('p'))
        
        # 特殊键需要使用不同名称
        self.plotter.add_key_event('plus', lambda: key_press('+'))
        self.plotter.add_key_event('minus', lambda: key_press('-'))


    def clear_changes(self):
        """清除所有颜色修改"""
        if 'RGB' in self.mesh.point_data:
            rgb = self.mesh.point_data['RGB']
            self.colors[:, 0:3] = rgb / 255.0
        else:
            self.colors[:] = np.ones((self.mesh.n_points, 4)) * 0.7
            self.colors[:, 3] = 1.0
            
        # 确保更新网格颜色
        self.mesh.point_data["colors"] = self.colors
        
        # 使用强化的渲染刷新方法
        self.refresh_render()
        
        self.modified_vertices = []
        print("已清除所有颜色修改")
    
    def save_mesh(self, custom_filename=None):
        """保存修改后的网格，可指定自定义文件名"""
        print(f"保存网格，已修改顶点数: {len(self.modified_vertices)}")
        
        # 获取保存路径
        if custom_filename is None:
            # 从输入文件名生成默认输出文件名
            base_name = os.path.splitext(os.path.basename(self.mesh_file))[0]
            default_output = f"{base_name}_modified.ply"
            
            # 询问用户是否使用默认文件名
            print(f"将保存为: {default_output}")
            user_input = input("按回车确认，或输入新文件名: ").strip()
            
            output_file = user_input if user_input else default_output
        else:
            output_file = custom_filename
    
        # 确保文件扩展名正确
        if not output_file.lower().endswith('.ply'):
            output_file += '.ply'

        # 使用直接文本方式创建标准PLY文件
        try:
            # 获取网格数据
            vertices = self.mesh.points
            faces = self.mesh.faces
            
            # 计算面的数量
            face_count = 0
            i = 0
            while i < len(faces):
                verts_per_face = faces[i]
                face_count += 1
                i += verts_per_face + 1
                
            # 将颜色转换为RGB 0-255整数格式
            rgb = np.round(self.colors[:, 0:3] * 255).astype(np.uint8)
            
            # 写入PLY文件
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入PLY头部
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write(f"element face {face_count}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # 写入顶点和颜色数据
                for i in range(len(vertices)):
                    v = vertices[i]
                    c = rgb[i]
                    f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
                
                # 写入面数据
                i = 0
                while i < len(faces):
                    verts_per_face = faces[i]
                    f.write(f"{verts_per_face}")
                    for j in range(1, verts_per_face + 1):
                        f.write(f" {faces[i+j]}")
                    f.write("\n")
                    i += verts_per_face + 1
                    
            print(f"已成功直接创建PLY文件: {output_file}")
            
            # 验证保存成功
            test_mesh = pv.read(output_file)
            print(f"验证文件: 包含 {test_mesh.n_points} 个点")
            
            # 检查颜色
            if 'red' in test_mesh.point_data or 'RGB' in test_mesh.point_data:
                print("验证成功: 颜色数据已成功保存")
            else:
                print("警告: 无法验证颜色数据，需手动检查")
                
            # # 记录修改的顶点
            # log_file = os.path.splitext(output_file)[0] + "_vertices.txt"
            # with open(log_file, 'w') as f:
            #     for v_id in self.modified_vertices:
            #         color = self.colors[v_id]
            #         f.write(f"{v_id}: [{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]\n")
            # print(f"已保存 {len(self.modified_vertices)} 个顶点的颜色修改记录到 {log_file}")
            
            # 标记为已保存
            self._saved = True
            
        except Exception as e:
            print(f"PLY保存失败: {e}")
            import traceback
            traceback.print_exc()
            
            print("尝试备用保存方法...")
            try:
                # 使用numpy和open3d直接保存
                import tempfile
                try:
                    import open3d as o3d
                    has_open3d = True
                except ImportError:
                    has_open3d = False
                
                if has_open3d:
                    # 转换为Open3D格式
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(self.mesh.points)
                    pcd.colors = o3d.utility.Vector3dVector(self.colors[:, :3])  # RGB部分
                    
                    # 保存为PLY格式
                    o3d.io.write_point_cloud(output_file, pcd)
                    print(f"使用Open3D成功保存颜色到: {output_file}")
                else:
                    # 使用NumPy结构化数组直接保存为PLY
                    # 创建带颜色的结构化数组
                    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                    
                    # 准备数据
                    points = self.mesh.points
                    colors = np.round(self.colors[:, :3] * 255).astype(np.uint8)
                    
                    # 创建结构化数组
                    data = np.empty(len(points), dtype=dtype)
                    data['x'] = points[:, 0]
                    data['y'] = points[:, 1]
                    data['z'] = points[:, 2]
                    data['red'] = colors[:, 0]
                    data['green'] = colors[:, 1]
                    data['blue'] = colors[:, 2]
                    
                    # 直接使用numpy-stl保存
                    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp:
                        temp_name = temp.name
                    
                    # 手动写入PLY文件
                    with open(temp_name, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(data)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write("end_header\n")
                        
                        for i in range(len(data)):
                            d = data[i]
                            f.write(f"{d['x']} {d['y']} {d['z']} {d['red']} {d['green']} {d['blue']}\n")
                            
                # 复制到目标位置
                import shutil
                shutil.copy(temp_name, output_file)
                os.unlink(temp_name)
                
                print(f"使用NumPy直接保存颜色到: {output_file}")
                
            except Exception as e3:
                print(f"所有保存方法均失败: {e3}")

    def save_screenshot(self):
        """保存当前视图的截图"""
        # 生成默认文件名
        base_name = os.path.splitext(os.path.basename(self.mesh_file))[0]
        default_output = f"{base_name}_screenshot.png"
        
        # 询问用户是否使用默认文件名
        print(f"将截图保存为: {default_output}")
        user_input = input("按回车确认，或输入新文件名: ").strip()
        
        output_file = user_input if user_input else default_output
        
        # 确保文件扩展名正确
        if not output_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_file += '.png'
        
        # 保存截图
        try:
            self.plotter.screenshot(output_file)
            print(f"已保存当前视图截图到: {output_file}")
        except Exception as e:
            print(f"保存截图失败: {e}")

    def toggle_batch_mode(self):
        """切换批量选择模式"""
        self.batch_mode = not self.batch_mode
        mode = "开启" if self.batch_mode else "关闭"
        print(f"批量选择模式: {mode} (当前半径: {self.selection_radius:.3f})")
        
        # 实时显示选择半径
        if hasattr(self, 'radius_indicator') and self.radius_indicator is not None:
            self.plotter.remove_actor(self.radius_indicator)
        
        if self.batch_mode and self.last_selected_point is not None:
            # 创建一个表示选择半径的球体
            center = self.mesh.points[self.last_selected_point]
            sphere = pv.Sphere(center=center, radius=self.selection_radius)
            self.radius_indicator = self.plotter.add_mesh(
                sphere,
                style='wireframe',
                color='white',
                opacity=0.3,
                render=True
            )

    def increase_selection_radius(self):
        """增加选择半径"""
        self.selection_radius *= 1.2
        print(f"增加选择半径至: {self.selection_radius:.3f}")

    def decrease_selection_radius(self):
        """减小选择半径"""
        self.selection_radius /= 1.2
        print(f"减小选择半径至: {self.selection_radius:.3f}")

    def refresh_render(self, immediate=False):
        """强化渲染刷新机制，确保颜色变化立即呈现，但视图不跳变"""
        # 保存完整的相机状态
        camera = self.plotter.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = self.plotter.camera.GetFocalPoint()
        view_up = self.plotter.camera.GetViewUp()
        view_angle = self.plotter.camera.GetViewAngle()
        clip_range = camera.GetClippingRange()
   
        # 确保颜色数据已更新到网格
        self.mesh.point_data["colors"] = self.colors
        
        # 更强制的刷新
        if hasattr(self.mesh, 'Modified'):
            self.mesh.Modified()  # 通知VTK网格已经修改
        
        # 通过多种方式强制刷新渲染
        if hasattr(self, 'actor'):
            self.actor.GetMapper().Update()
        
        # 重新应用数据
        self.mesh.GetPointData().Modified()
        
        # 移除不兼容的参数，使用更通用的方法刷新
        if immediate:
            # 使用update但只有reset_camera参数
            self.plotter.update(reset_camera=False)
            # 直接调用渲染窗口的渲染方法以立即更新
            if hasattr(self.plotter.renderer, 'GetRenderWindow'):
                render_window = self.plotter.renderer.GetRenderWindow()
                if render_window:
                    render_window.Render()
        else:
            self.plotter.update(reset_camera=False)
            self.plotter.render()
    
        # 确保完全恢复相机状态
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)
        camera.SetViewAngle(view_angle)
        camera.SetClippingRange(clip_range)


    def color_points_in_radius(self, center_point_id):
        """对指定半径内的点应用当前颜色"""
        center = self.mesh.points[center_point_id]
        distances = np.linalg.norm(self.mesh.points - center, axis=1)
        
        # 选择半径内的所有点
        points_in_radius = np.where(distances <= self.selection_radius)[0]
        
        # 应用当前颜色
        for pid in points_in_radius:
            self.colors[pid] = self.current_color
            if pid not in self.modified_vertices:
                self.modified_vertices.append(pid)
        
        # 使用强化的渲染刷新方法
        self.refresh_render()
        print(f"已修改 {len(points_in_radius)} 个顶点的颜色")

    def save_state_for_undo(self):
        """保存当前状态以支持撤销"""
        # 只保存最近10个状态以避免内存占用过大
        if len(self.color_history) >= 10:
            self.color_history.pop(0)
        self.color_history.append(self.colors.copy())

    def undo_last_change(self):
        """撤销上一次颜色修改"""
        if not self.color_history:
            print("没有可撤销的操作")
            return
            
        # 恢复到上一个状态
        self.colors = self.color_history.pop()
        self.mesh.point_data["colors"] = self.colors
        
        # 使用强化的渲染刷新方法
        self.refresh_render()
        print("已撤销上一次修改")

    def pick_color_from_point(self):
        """从已有顶点获取颜色"""
        if self.last_selected_point is None:
            print("请先选择一个点")
            return
            
        # 获取所选点的颜色
        color = self.colors[self.last_selected_point].copy()
        self.current_color = color
        print(f"已从顶点 {self.last_selected_point} 获取颜色: [{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]")

    def show_selected_point(self, point_id):
        """显示当前选中的点，增强视觉反馈"""
        # 如果之前有显示的点，先移除
        if hasattr(self, 'selected_point_actor') and self.selected_point_actor is not None:
            self.plotter.remove_actor(self.selected_point_actor)

        # 显示当前选中的点
        point = self.mesh.points[point_id]
        sphere = pv.Sphere(center=point, radius=self.selection_radius*0.5)
        self.selected_point_actor = self.plotter.add_mesh(
            sphere, 
            color='yellow', 
            style='wireframe', 
            line_width=2,
            render=True
        )

    def toggle_facets(self):
        """切换三角面片的显示/隐藏"""
        self.show_facets = not self.show_facets
        
        # 移除当前的网格显示
        self.plotter.remove_actor(self.actor)
        
        if self.show_facets:
            # 显示三角面片细节 - 使用线框模式
            self.actor = self.plotter.add_mesh(
                self.mesh, 
                scalars="colors", 
                rgb=True,
                style='wireframe',
                line_width=1,
                show_edges=True
            )
            print("已开启三角面片显示模式")
        else:
            # 正常表面显示模式
            self.actor = self.plotter.add_mesh(
                self.mesh, 
                scalars="colors", 
                rgb=True,
                show_edges=False
            )
            print("已关闭三角面片显示模式")
        
        # 强制刷新渲染但保持相机位置
        self.refresh_render(immediate=True)

    def toggle_fullscreen(self):
        """切换全屏/窗口模式"""
        if hasattr(self.plotter, 'ren_win') and self.plotter.ren_win:
            is_fullscreen = bool(self.plotter.ren_win.GetFullScreen())
            self.plotter.ren_win.SetFullScreen(not is_fullscreen)
            status = "窗口" if is_fullscreen else "全屏"
            print(f"已切换到{status}模式")

    def quit(self):
        """安全退出程序"""
        print("正在退出程序...")
        
        # 清理资源
        if hasattr(self, 'selected_point_actor') and self.selected_point_actor is not None:
            self.plotter.remove_actor(self.selected_point_actor)
        
        if hasattr(self, 'radius_indicator') and self.radius_indicator is not None:
            self.plotter.remove_actor(self.radius_indicator)
        
        # 关闭渲染窗口
        self.plotter.close()
        
        # 提示用户是否保存
        if self.modified_vertices and not self._saved:
            print("注意：您有未保存的修改。")
            user_input = input("是否保存修改？(y/n): ")
            if user_input.lower() == 'y':
                self.save_mesh()
        
        sys.exit(0)

    def run(self):
        """运行交互式编辑器"""
        self.setup()
        
        print("\n==== 顶点颜色编辑器 (扩展版) ====")
        print("- 点击网格上的顶点修改颜色")
        print("- 颜色选择快捷键:")
        print("  r: 红色   g: 绿色   b: 蓝色")
        print("  y: 黄色   c: 青色   m: 洋红色")
        print("  w: 白色   k: 黑色")
        print("- 功能键:")
        print("  a: 切换批量模式     +/-: 调整选择半径")
        print("  i: 获取点的颜色     z: 撤销上一步")
        print("  x: 清除所有修改     s: 保存修改网格")
        print("  f: 显示/隐藏三角面片 p: 保存当前视图截图")
        print("  z: 切换全屏/窗口模式 q: 退出")
        
        self.plotter.show()


if __name__ == "__main__":
    # 默认文件路径
    mesh_file = "AD8EQEUR_upper_colored.ply"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    
    if not os.path.exists(mesh_file):
        print(f"错误: 文件 '{mesh_file}' 不存在")
        sys.exit(1)
    
    # 创建并运行编辑器
    editor = VertexColorEditor(mesh_file)
    editor.run()