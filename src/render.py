import warp as wp
import numpy as np
import imageio

from .camera import Camera
from .object import Object

@wp.kernel
def raycast(
    mesh: wp.uint64,
    ray_origin: wp.vec3,
    ray_direction: wp.array2d(dtype=wp.vec3),
    image: wp.array2d(dtype=wp.vec3),
):
    j, i = wp.tid()

    t = float(0.)       # hit distance
    u = float(0.)       # hit face barycentric u
    v = float(0.)       # hit face barycentric v
    sign = float(0.)    # hit face sign
    n = wp.vec3(0.)     # hit face normal
    f = int(0)          # hit face index

    color = wp.vec3(0.)

    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction[j, i], 1.e6, t, u, v, sign, n, f):
        color = n * .5 + wp.vec3(.5, .5, .5)

    image[j, i] += color

class Render:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = Camera()

    def add_object(self, object: Object):
        self.objects.append(object)

    def add_light(self, light: wp.vec3):
        self.lights.append(light)

    def render(self):
        # if len(self.objects) == 0 or len(self.lights) == 0:
        #     raise ValueError("No object or light specified")
        object = Object(mesh_path="data/cylinder.obj")
        width = self.camera.image_width
        height = self.camera.image_height
        ray_origin = self.camera.rays.origin
        ray_direction = self.camera.rays.direction[0]
        image = self.camera.image
        wp.launch(
            kernel=raycast,
            dim=(height, width),
            inputs=[object.mesh.id, ray_origin, ray_direction],
            outputs=[image],
        )
        image_np = np.flipud(np.clip(image.numpy() * 255, 0, 255).astype(np.uint8))
        imageio.imwrite(f"output/output.png", image_np)
