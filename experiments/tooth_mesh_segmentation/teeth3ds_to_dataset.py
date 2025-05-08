import os
import json

def traverse_teeth3ds(directory):
    dest_dir = directory+"_dataset"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)   
        
    label_dir = os.path.join(dest_dir, "labels")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    mesh_dir = os.path.join(dest_dir, "meshes")
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    
    """遍历 Teeth3DS+ 文件夹下的子目录并打印文件路径"""
    for root, dirs, files in os.walk(directory):
        # print(f"当前目录: {root}")
        for file in files:
            # print(f"文件: {os.path.join(root, file)}")
            # print("Root",root)
            if file.endswith(".obj"):
                # 处理 obj 文件
                obj_filepath = os.path.join(root, file)
            
                json_filepath = obj_filepath[0:-4] + ".json"
                if not os.path.exists(json_filepath):
                    print(f"json file not found: {json_filepath}")
                    continue
                
                # 复制到目标目录
                os.system(f"cp {obj_filepath} {mesh_dir}")
                
                with open(json_filepath, "r") as f:
                    annotation = json.load(f)
                txt_filepath = os.path.join(label_dir, file[0:-4]+".txt")                
                print(f"txt file: {txt_filepath}")
                with open(txt_filepath, "w") as f:
                    for i in range(len(annotation["labels"])):
                        label = annotation["labels"][i]
                        f.write(f"{label}\n")
                     
if __name__ == "__main__":
    # Teeth3DS+ 文件夹路径
    teeth3ds_dir = "/home/jelly/Datasets/Teeth3DS+"
        
    traverse_teeth3ds(teeth3ds_dir)

