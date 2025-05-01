import warp as wp
import numpy as np
import open3d as o3d

class Mesh:
    def __init__(self, vertices=None, indices=None, mesh_path=None):
        self.vertices = None
        self.indices = None
        
        if vertices is not None and indices is not None:
            self._load_from_data(vertices, indices)
        elif mesh_path is not None:
            self._load_from_mesh(mesh_path)
        else:
            raise ValueError("Must provide vertices/indices or mesh_path.")
        
        self.data = wp.Mesh(points=self.vertices, indices=self.indices)
        self.id = self.data.id
    
    def _load_from_data(self, vertices_np, indices_np):
        ''' Load from numpy array '''
        self.num_vertices = len(vertices_np)
        self.num_indices = len(indices_np)
        self.vertices = wp.from_numpy(vertices_np, shape=(self.num_vertices,), dtype=wp.vec3)
        self.indices = wp.from_numpy(indices_np, shape=(self.num_indices*3,), dtype=wp.int32)
    
    def _load_from_mesh(self, mesh_path):
        ''' Load from triangle mesh '''
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices_np = np.asarray(mesh.vertices)
        indices_np = np.asarray(mesh.triangles)
        self.num_vertices = len(vertices_np)
        self.num_indices = len(indices_np)
        self.vertices = wp.from_numpy(vertices_np, shape=(self.num_vertices,), dtype=wp.vec3)
        self.indices = wp.from_numpy(indices_np, shape=(self.num_indices*3,), dtype=wp.int32)
