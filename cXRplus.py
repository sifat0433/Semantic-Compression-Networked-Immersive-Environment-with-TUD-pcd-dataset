import numpy as np
import GC as g
import open3d as o3d


#global lookup tables creation function

def lookuptable(bound_min,bound_max,grid_step):
    grid_space = np.mgrid[bound_min:(bound_max+grid_step):grid_step, bound_min:(bound_max+grid_step):grid_step, bound_min:(bound_max+grid_step):grid_step].reshape(3, -1).T

    #global point cloud PC_g
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_space)

    # global voxel frame V_g
    voxel_frame = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.04) # Standard voxel size variation 3-5cm

    #color assignin to each voxels
    colorlookup = np.asarray([pt.grid_index for pt in voxel_frame.get_voxels()])
    colorlookup= colorlookup.tolist()
    colorlookup.sort()

    return colorlookup, voxel_frame, len(grid_space)

#upper and lower bound of the environment
upper_bound = 3.9
lower_bound = -2.8
gridStep = 0.02 #step size for 3D grid creation


Clookup, Gvoxel, grid = g.funcCompressPC(lower_bound,upper_bound,gridStep)



def voxel_select(queries, global_voxel): # obtainin the poisition of points withing voxels
    v= np.asarray([global_voxel.get_voxel(pt) for pt in queries])
    return v

def get_color(occ_voxel,Clookup): # Assiging colors to the points for corresponding voxels
    colorlist =[]
    ylen = max(Clookup)[1]+1
    zlen = max(Clookup)[2]+1
    for pt in occ_voxel:
        colorlist.append([((pt[0]*ylen+pt[1])*zlen+pt[2])])
    return colorlist

def reconstruct(sent_voxel,global_voxel):
    voxel_centr = np.array([global_voxel.get_voxel_center_coordinate(pt) for pt in sent_voxel]) #obtaining center ponits of the voxels
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(voxel_centr) #creating a temporary point cloud with center ponits of the voxels
    re_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(p, voxel_size=0.04) #creating voxel frame from the center ponits point cloud
    o3d.visualization.draw_geometries([re_voxel]) #reconstructed voxel frame visualization



number_of_objects = 4 # small, medium, large, very large objects
number_of_frames = 14 #14 frames means 14 pcd files

for ob in range(number_of_objects):
    for t in range(number_of_frames):

        # Client VNF
        # Input
        pcd = o3d.io.read_point_cloud('dataset/'+str(ob+1) +'/('+ str(t + 1) + ').pcd') # reading point cloud (x_t) for each frame of each object
        pcd_points = np.asarray(pcd.points) # obtaining all the point locations with point cloud (x_t)

        # Encoding (Step 1: online) + Compression (Step 2: online)
        # obtaining unique color list assigned to the point cloud (x_t)
        occ_voxels = voxel_select(pcd_points,Gvoxel)
        occ_voxels = np.unique(occ_voxels, axis=0)
        colors = get_color(occ_voxels,Clookup)

        # Forwarding (Step 3: online)
        # color tansmission over the network.
        # UDP/TCP can be use or Transmission time can be calculated from number of bits needed to represent the unique colorlist.
        # Calculating parameters are given in the paper.


        # Server VNF

        # Decoding (Step 4: online)
        decoded_voxel=[]
        for vc in colors:
            decoded_voxel.append(Clookup[vc[0]])
        decoded_voxel = np.array(decoded_voxel)

        # volume reconstruction (Step 5: online)
        reconstruct(decoded_voxel,Gvoxel)

