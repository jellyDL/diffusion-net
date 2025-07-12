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
        
        # 添加原图而不添加标签
        images.append(img)
    
    # 计算网格尺寸
    rows = (len(images) + cols - 1) // cols  # 向上取整
    
    # 确保所有图片尺寸一致
    max_img_height = max(img.shape[0] for img in images)
    max_img_width = max(img.shape[1] for img in images)
    
    # 计算总的画布尺寸 - 增加底部标签区域和额外留白
    total_width = cols * max_img_width + (cols - 1) * spacing
    # 增加标签区域的高度，提供更多的底部留白
    bottom_padding = label_height + 50  # 增加额外的50像素留白
    total_height = rows * max_img_height + (rows - 1) * spacing + bottom_padding
    
    # 创建最终的合并图像 - 增加底部标签区域
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
    
    # 将numpy数组转换为PIL图像
    if merged_image.dtype == np.float32 or merged_image.dtype == np.float64:
        pil_image = Image.fromarray((merged_image * 255).astype(np.uint8))
    else:
        pil_image = Image.fromarray(merged_image)
    
    # 添加底部标签
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # 在底部添加标签，位置上移以留出更多底部空间
    label_y = rows * max_img_height + (rows - 1) * spacing + (label_height // 4)
    for col, label in enumerate(label_list[:cols]):
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        
        # 计算每列的标签位置
        label_x = col * (max_img_width + spacing) + (max_img_width - text_width) // 2
        
        # 添加文字
        draw.text((label_x, label_y), label, fill='black', font=font)
    
    # 降低饱和度
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.5)  # 0.5表示保留50%的饱和度
    
    # 保存图像
    pil_image.save(output_path)

def add_and_sort_images(image_names_sorted, image_path, sorted_indices):
    
    end_wish_list = ["DGCNN", "SimpSegNet", "Geo-Net", "CurSegNet", "TSegLab", "Ours", "GT"]
    # 按照end_wish_list的顺序，依次从image_path获取到图片，添加到image_names_sorted
    for name in end_wish_list:
        # 获取当前目录下所有包含name的png文件
        image_names = [f for f in os.listdir(image_path) if f.find(name)!=-1 and f.endswith('.png')]
        if not image_names:
            print(f"没有找到包含 '{name}' 的图片，请检查目录: {image_path}")
            # append 一张空白黑图
            empty_image = np.zeros((200, 200, 3), dtype=np.uint8)
            empty_image_pil = Image.fromarray(empty_image)
            empty_image_pil.save(os.path.join(image_path, f"{name}_empty.png"))
            image_names_sorted.append(os.path.join(image_path, f"{name}_empty.png"))
        else:
            image_names_sorted.append(os.path.join(image_path, image_names[0]))

if __name__ == "__main__":
    
    # 图像列的label名称
    label_list =[]
    name_list = ["DGCNN", "SimpSegNet", "Geo-Net", "CurSegNet", "TSegLab", "Ours", "Ground Truth"]
    for name in name_list:
        label_list.append(name)
        
    image_files_sorted = []
    add_and_sort_images(image_files_sorted,  "render_pics_1", sorted_indices=[0,6,2,5,4,3,1])
    add_and_sort_images(image_files_sorted,  "render_pics_2", sorted_indices=None)
    # add_and_sort_images(image_files_sorted,  "render_pics_3", sorted_indices=None)
    # add_and_sort_images(image_files_sorted,  "render_pics_4", sorted_indices=None)
    
    
    output_file = "merged_rendered_images.png"
    spacing = 2  # 减小图片之间的间距，从5降低到2
    label_height = 160  # 标签区域的高度
    font_size = 120  # 标签字体大小
    
    merge_images_grid(image_files_sorted, label_list, label_height, font_size,
                    output_file, spacing, cols=len(label_list))
        
        
    print(f"已合并 {len(image_files_sorted)} 张图片为 {len(image_files_sorted) / len(label_list)} 行,  {len(label_list)} 列网格，间距为 {spacing} 像素，输出文件: {output_file}")
