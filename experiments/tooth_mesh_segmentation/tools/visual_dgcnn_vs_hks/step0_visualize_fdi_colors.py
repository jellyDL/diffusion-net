import json
import matplotlib.pyplot as plt

def load_fdi_colors(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 兼容dict、list等多种结构
    if isinstance(data, dict):
        # 尝试将dict的values合并为list
        data = list(data.values())
    # 跳过空数据
    if not data:
        return [], []
    # 如果data是嵌套list，展开
    if isinstance(data, list) and isinstance(data[0], list):
        flat_data = []
        for sublist in data:
            flat_data.extend(sublist)
        data = flat_data
    labels = []
    colors = []
    raw_colors = []
    for item in data:
        labels.append(item['label'])
        color = item['color']
        raw_colors.append(color)
        if isinstance(color, list):
            color = [c/255 if max(color) > 1 else c for c in color]
        colors.append(color)
    return labels, colors, raw_colors

def visualize_colors(labels, colors, raw_colors):
    fig, ax = plt.subplots(figsize=(3, len(labels)*0.28))  # 增加高度
    for i, (label, color, raw_color) in enumerate(zip(labels, colors, raw_colors)):
        ax.barh(i, 1, color=color, edgecolor='black')
        ax.text(1.05, i, f"{label}  {raw_color}", va='center', fontsize=8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 2)
    ax.set_title("FDI Label-Color 对应关系")
    plt.tight_layout()
    plt.show()

def main():
    json_path = "fdi_number.json"
    labels, colors, raw_colors = load_fdi_colors(json_path)
    visualize_colors(labels, colors, raw_colors)

if __name__ == "__main__":
    main()
