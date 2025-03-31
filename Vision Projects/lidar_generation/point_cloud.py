import numpy as np
import open3d as o3d

def depth_to_point_cloud(depth_map, intrinsic_matrix):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map.flatten()
    x = (i.flatten() - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (j.flatten() - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
    points = np.vstack((x, y, z)).T
    points = points[z > 0]  # Remove points with zero depth
    return points

def visualize_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])
