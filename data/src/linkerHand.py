import torch
import pytorch_kinematics as pk
from trimesh import Trimesh
import os
from spatialmath import SE3,SO3
import trimesh
import open3d as o3d

current_dir = os.path.dirname(__file__)

def twist_2_SE3(tw)->SE3:
    twist = []
    for i in range(6):
        twist.append(float(tw[i]))

    T = SE3(twist[0], twist[1], twist[2])
    T = T * SE3(SO3.RPY(twist[5], twist[4], twist[3]))
    return T

class LINK:
    def __init__(self, name:str, mesh:Trimesh, chain:pk.Chain, side:str, num_sample:int=500):
        '''### link related information
        `name`: link name, str  
        `mesh`: link trimesh  
        `chain`: kinematic chain that the link belongs to (either left hand or right hand)
        '''
        self.name = name
        self.mesh = mesh
        self.chain = chain
        self.side = side
        self.frame_idx = self.chain.get_frame_indices(name) # frame index
        
        self.T_base2link = torch.eye(4)
    
    def update_base2link_tf(self, q:torch.Tensor):
        '''FK and update T_base2link'''
        ret = self.chain.forward_kinematics(q, frame_indices=[self.frame_idx])
        mat = ret[self.name].get_matrix() # get transform matrix (1,4,4)
        self.T_base2link = mat.squeeze(dim=0)

class linkerHand():
    '''
    ### 16-dof order of linkerHand
    thumb: rotation, abduction, proximal bent, distal bent
    index: spread, proximal bent, distal bent
    middle: spread, proximal bent, distal bent
    ring: spread, proximal bent, distal bent
    little: spread, proximal bent, distal bent'''
    dof = 16
    full_dof = 16+5
    lowerLimit = ()
    upperLimit = ()
    anchorPoint = {}
    
    def __init__(self):
        self.q = torch.zeros(self.dof) #joint angles
        self.link_info = {} # key: link name; value: LINK object
    
    @staticmethod
    def q2q_pk(q:torch.Tensor, full_dof:int=21) -> torch.Tensor:
        '''direct mapping 16 dof finger joint angles to linker hand for pytorch_kinematics
        parameter: ``q`` Tensor(16,), active dof
        return: ``q_linkerHand`` Tensor(21,), full dof
        '''
        q_linkerHand = torch.zeros(full_dof)
        mimic = {
            4: (3, 1.34/1.32), # thumb passive dof: (index, scale)
            8: (7, 1.48/1.36), # index
            12: (11, 1.48/1.36), # middle
            16: (15, 1.48/1.36), # ring
            20: (19, 1.48/1.36), # little
        }
        full_joint = {
            0:q[0], 1:q[1], 2:q[2], 3:q[3],
            5:q[4], 6:q[5], 7:q[6],
            9:q[7], 10:q[8], 11:q[9],
            13:q[10], 14:q[11], 15:q[12],
            17:q[13], 18:q[14], 19:q[15]
        } # key: joint index; value: joint angle
        for key, value in mimic.items():
            full_joint.update({key: full_joint[value[0]] * value[1]})

        for i in range(full_dof):
            q_linkerHand[i] = full_joint[i]
        return q_linkerHand
    
    def update_hand_state(self, q:torch.Tensor):
        '''need to be called after each iteration
        update joint angles and link base2link tf matrix'''
        self.q = q
        q_pk = self.q2q_pk(q)
        for link_name, link in self.link_info.items():
            link.update_base2link_tf(q_pk)
    
    def get_mesh_o3d(self, grasp, color=[166/255, 209/255, 234/255]):
        """get hand mesh from the grasp parameters, and save the o3dTriangleMesh
        
        Args
        -------
        grasp (torch.Tensor)
            grasp parameters, shape (3+3+16,)
            
        Return
        -------
        meshes (dict)
            dict of o3d.geometry.TriangleMesh; key is the link name
        """
        T_wrist = twist_2_SE3(grasp[0:6]).A
        
        self.update_hand_state(grasp[6:22])
        
        meshes = []
        self.o3dTriangleMesh = {}
        for name, link in self.link_info.items():
            mesh_original = link.mesh
            T_base2link = link.T_base2link.numpy()
            vertices = mesh_original.vertices
            triangles = mesh_original.faces

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            # mesh.paint_uniform_color([0/255, 174/255, 239/255]) 
            mesh.transform(T_wrist @ T_base2link)# transform the mesh to the world frame
            meshes.append(mesh)
            self.o3dTriangleMesh[name] = mesh
        return meshes
        
class linkerHandL(linkerHand):
    '''### linkerHand left hand'''
    urdf_path = os.path.join(current_dir, 'URDF/linkerHand/L20_8_l_s/urdf/L20_8_l_s.urdf')
    meshes_path = os.path.join(current_dir, 'URDF/linkerHand/L20_8_l_s/meshes')
    side = 'left'
    robot_pk = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

    ## full-dof joint limit
    lowerLimit = (-1.05, 0.0, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0)
    upperLimit = (0.49, 1.45, 0.84, 1.32, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36)
   
    def __init__(self, q=torch.zeros(16)):
        super().__init__()
        self.q = q #joint angles
        self.link_info = {} # key: link name; value: LINK object
        for link_pk in self.robot_pk.get_links():
            if link_pk.visuals[0].geom_type == 'mesh':
                mesh_path = link_pk.visuals[0].geom_param[0]
                mesh_path = os.path.join(os.path.dirname(__file__), mesh_path)
                link_mesh = trimesh.load(mesh_path, force='mesh')
                link_name = link_pk.name
                link = LINK(link_name, link_mesh, chain=self.robot_pk, side=self.side)
                link.update_base2link_tf(self.q2q_pk(q))
                self.link_info[link_name] = link

class linkerHandR(linkerHand):
    '''### linkerHand right hand'''
    urdf_path = os.path.join(current_dir, 'URDF/linkerHand/L20_8_r_s1/urdf/L20_8_r_s1.urdf')
    meshes_path = os.path.join(current_dir, 'URDF/linkerHand/L20_8_r_s1/meshes')
    side = 'right'
    robot_pk = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

    ## full-dof joint limit
    lowerLimit = (-0.49, -1.45, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0, -0.26, 0.0, 0.0)
    upperLimit = (1.05, 0.0, 0.84, 1.32, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36, 0.26, 1.66, 1.36)
    
    def __init__(self, q=torch.zeros(16)):
        super().__init__()
        self.q = q #joint angles
        self.link_info = {} # key: link name; value: LINK object
        for link_pk in self.robot_pk.get_links():
            if link_pk.visuals[0].geom_type == 'mesh':
                mesh_path = link_pk.visuals[0].geom_param[0]
                mesh_path = os.path.join(current_dir, mesh_path)
                link_mesh = trimesh.load(mesh_path, force='mesh')
                link_name = link_pk.name
                link = LINK(link_name, link_mesh, chain=self.robot_pk, side=self.side)
                link.update_base2link_tf(self.q2q_pk(q))
                self.link_info[link_name] = link