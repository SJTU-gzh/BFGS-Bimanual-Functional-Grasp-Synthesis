import open3d as o3d
import os

mesh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'template/mano_grasp_template.ply')
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])