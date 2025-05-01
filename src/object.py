from .mesh import Mesh
from .material import Meterial

class Object:
    def __init__(self, vertices=None, indices=None, mesh_path=None):
        self.mesh = Mesh(vertices=vertices, indices=indices, mesh_path=mesh_path)
        self.meterial = Meterial()
