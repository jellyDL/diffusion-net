"""
This program reads an obj file of a 3D model and an annotation json file,
and displays the model with colors corresponding to the labels assigned
to each vertex.

Usage: python show_annoation_for_teeth3ds.py <obj_filepath>

Args:
    obj_filepath: Path to the obj file. It is assumed that a json file
                  with the same name, written in teeth3ds format,
                  exists in the same directory.
"""
import json
import os
import sys
import open3d as o3d
import trimesh


def main(argv):
    if len(argv) < 2:
        dataset_path = os.path.join("/Users/jelly/Desktop/Teeth3DS+/upper")
    else:
        obj_folder = argv[1]
        dataset_path = os.path.join("Users/jelly//Desktop/Teeth3DS+/upper", obj_folder)
    
    print(f"dataset_path: {dataset_path}")
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        print(f"Listing files in dataset path: {folder_path}")
        for root, dirs, files in os.walk(folder_path):
            obj_filepath = None
            json_filepath = None
            
            for file in files:
                if file.endswith(".obj"):
                    obj_filepath = os.path.join(root, file)
                elif file.endswith(".json"):
                    json_filepath = os.path.join(root, file)
            
            if not obj_filepath or not json_filepath:
                print(f"Skipping folder {folder_path} as it does not contain required files.")
                continue

            try:
                # load fdi number color json file
                fdi_json_filepath = r"./fdi_number.json"
                print(f"fdi_json_filepath: {fdi_json_filepath}")
                with open(fdi_json_filepath, "r") as f:
                    fdi_numbers = json.load(f)

                # because open3d removes vertices when loading,
                # load with trimesh and convert to open3d
                trimesh_mesh = trimesh.load(obj_filepath)
                print(f"the number of vertices: {len(trimesh_mesh.vertices)}")
                print(f"the number of faces: {len(trimesh_mesh.faces)}")

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
                mesh.compute_vertex_normals()

                # load annotation json file based on teeth3ds format
                json_filepath = os.path.splitext(obj_filepath)[0] + ".json"
                with open(json_filepath, "r") as f:
                    annotation = json.load(f)
                print(f"the number of vertices: {len(mesh.vertices)}")
                colors = []
                for i in range(len(mesh.vertices)):
                    label = annotation["labels"][i]
                    founds = [element for element in fdi_numbers["fdi_colors"]
                            if element["label"] == label]
                    if founds:
                        colors.append([
                            founds[0]["color"][0] / 255.0,
                            founds[0]["color"][1] / 255.0,
                            founds[0]["color"][2] / 255.0
                        ])
                    else:
                        colors.append([1.0, 1.0, 1.0])  # default color

                # set vertex colors
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                # 保存为 PLY 文件
                ply_filepath = os.path.splitext(obj_filepath)[0] + "_colored.ply"
                o3d.io.write_triangle_mesh(ply_filepath, mesh)
                print(f"Colored mesh saved to: {ply_filepath}")

                # 可视化网格
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(mesh)
                vis.update_renderer()
                vis.run()
                vis.destroy_window()
                # return 0

            except Exception as e:
                print(e)
                return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))