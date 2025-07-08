import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def merge_images_grid(image_files, label_list, label_height, font_size, output_path, spacing=5, cols=3):
    """
    将多张图片按网格排布并保存为一张整图。

    :param image_path: 图片文件夹路径
    :param image_names: 图片文件名列表
    :param output_path: 输出文件路径
    :param spacing: 图片之间的间距，单位为像素
    :param cols: 每行图片数量
    """
    images = []
    
    for i, image_file in enumerate(image_files):
        print("### image_file", image_file)
        img = plt.imread(image_file)
        # 处理不同通道数的图片
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]  # 只取RGB通道
        elif img.ndim == 2:  # 灰度图
            img = np.stack([img] * 3, axis=2)  # 转换为RGB
        
        # 创建带有标签区域的新图像
        new_height = img.shape[0] + label_height
        new_width = img.shape[1]
        
        # 创建新的图像数组，上方为标签区域，下方为原图
        if img.dtype == np.float32 or img.dtype == np.float64:
            new_img = np.ones((new_height, new_width, 3), dtype=img.dtype)  # 白色背景
            new_img[label_height:, :] = img  # 将原图放在下方
            img_pil = Image.fromarray((new_img * 255).astype(np.uint8))
        else:
            new_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 白色背景
            new_img[label_height:, :] = img  # 将原图放在下方
            img_pil = Image.fromarray(new_img)
        
        # 添加标签到上方区域
        draw = ImageDraw.Draw(img_pil)
        # label = f"({chr(97 + i)})"  # (a), (b), (c), ...
        label = label_list[i % cols] 
        
        # 尝试使用系统字体，字体更大
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 在标签区域居中添加文字
        x = (new_width - text_width) // 2
        y = (label_height - text_height) // 2
        
        if i < cols:
            draw.text((x, y), label, fill='black', font=font)
        
        # 转换回numpy数组
        img_with_label = np.array(img_pil)
        images.append(img_with_label)
    
    # 计算网格尺寸
    rows = (len(images) + cols - 1) // cols  # 向上取整
    
    # 确保所有图片尺寸一致
    max_img_height = max(img.shape[0] for img in images)
    max_img_width = max(img.shape[1] for img in images)
    
    # 计算总的画布尺寸
    total_width = cols * max_img_width + (cols - 1) * spacing
    total_height = rows * max_img_height + (rows - 1) * spacing
    
    # 创建最终的合并图像
    if images[0].dtype == np.float64 or images[0].dtype == np.float32:
        merged_image = np.ones((total_height, total_width, 3), dtype=np.float32)
    else:
        merged_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 将每张图片放置到网格中
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        # 计算位置
        y_start = row * (max_img_height + spacing)
        x_start = col * (max_img_width + spacing)
        
        # 放置图片
        h, w = img.shape[:2]
        merged_image[y_start:y_start + h, x_start:x_start + w] = img
    
    # 保存合并后的图像
    # 将numpy数组转换为PIL图像以调整饱和度
    if merged_image.dtype == np.float32 or merged_image.dtype == np.float64:
        pil_image = Image.fromarray((merged_image * 255).astype(np.uint8))
    else:
        pil_image = Image.fromarray(merged_image)
    
    # 降低饱和度
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.5)  # 0.7表示保留70%的饱和度
    
    # 保存图像
    pil_image.save(output_path)

def add_and_sort_images(image_names_sorted, image_path, sorted_indices):
    
    image_names = [f for f in os.listdir(image_path) if f.startswith('render_') and f.endswith('.png')]
    # 获取当前目录下所有以render_开头的png文件
    for index, image_name in enumerate(image_names):
        if sorted_indices :
            image_names_sorted.append(os.path.join(image_path, image_names[sorted_indices[index]]))
        else:
            image_names_sorted.append(os.path.join(image_path, image_name))
            
if __name__ == "__main__":
    
    # 图像列的label名称
    label_list =[]
    name_list = ["Raw Mesh", "DGCNN", "PointNext", "Geo-Net", "CrossTooth", "Ours", "Ground Truth"]
    for name in name_list:
        label_list.append(name)
        
    image_files_sorted = []
    add_and_sort_images(image_files_sorted,  "render_pics", sorted_indices=[5,4,0,1,3,2,6])
    add_and_sort_images(image_files_sorted,  "render_pics_2", sorted_indices=None)
    add_and_sort_images(image_files_sorted,  "render_pics_3", sorted_indices=None)
    add_and_sort_images(image_files_sorted,  "render_pics_4", sorted_indices=None)
    
    
    output_file = "merged_rendered_images.png"
    spacing = 50 # 图片之间的间距，单位为像素
    label_height = 180  # 标签区域的高度
    font_size = 80  # 标签字体大小
    
    merge_images_grid(image_files_sorted, label_list, label_height, font_size,
                    output_file, spacing, cols=len(label_list))
        
        
    print(f"已合并 {len(image_files_sorted)} 张图片为 {len(image_files_sorted) / len(label_list)} 行,  {len(label_list)} 列网格，间距为 {spacing} 像素，输出文件: {output_file}")
