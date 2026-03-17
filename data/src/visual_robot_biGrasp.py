import pickle
from linkerHand import linkerHandL, linkerHandR
import open3d as o3d
import torch
import os


def main(instance:str):
    '''
    visualize the biGrasp and object mesh for the specific instance
    '''
    graspL_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'biGrasp/{instance}/lhand_grasp.pkl')
    graspR_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'biGrasp/{instance}/rhand_grasp.pkl')
    obj_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'obj_contactMesh/{instance}/object_contact_mesh.ply')

    handL = linkerHandL()
    handR = linkerHandR()
    with open(graspL_path, 'rb') as f:
        graspL = pickle.load(f)
    with open(graspR_path, 'rb') as f:
        graspR = pickle.load(f)
    handL_meshes = handL.get_mesh_o3d(torch.from_numpy(graspL))
    handR_meshes = handR.get_mesh_o3d(torch.from_numpy(graspR))
    obj_mesh = o3d.io.read_triangle_mesh(obj_path)
    obj_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(handL_meshes+handR_meshes+[obj_mesh])
    
if __name__ == '__main__':
    instance = '115_180_8a23e8ae357fa2b71920da6870de352'
    main(instance)