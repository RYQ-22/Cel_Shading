import warp as wp

from .ray import Rays

@wp.kernel
def compute_rays(
    origin: wp.vec3,
    pixel00_loc: wp.vec3,
    pixel_delta_u: wp.vec3,
    pixel_delta_v: wp.vec3,
    direction: wp.array2d(dtype=wp.vec3),
):
    j, i = wp.tid()
    pixel_loc = pixel00_loc + pixel_delta_u * float(i) + pixel_delta_v * float(j)
    direction[j, i] = wp.normalize(pixel_loc - origin)

class Camera:
    def __init__(self):
        ########## Init Config ##########
        self.aspect_ratio = 1.          # Ratio of image width over height
        self.image_width = 1000          # Rendered image width
        self.samples_per_pixel = 1      # Count of random samples per pixel
        self.max_depth = 1              # Max number of ray bounces into scene

        self.vfov = 50.                     # Vertical view angle
        self.center = wp.vec3(4., 4., 4.)   # Point camera is looking from
        self.lookat = wp.vec3(0., 0., 0.)   # Point camera is looking at
        self.vup = wp.vec3(0., 1., 0.)      # Up direction in camera's view

        self.image_height = 0.              # Rendered image height
        self.pixel_samples_scale = 0.       # Color scale factor for a sum of pixel samples
        self.pixel00_loc = wp.vec3(0.)      # Location of pixel (0, 0)
        self.pixel_delta_u = wp.vec3(0.)    # Offset to pixel to the right
        self.pixel_delta_v = wp.vec3(0.)    # Offset to pixel below
        self.u = wp.vec3(0.)
        self.v = wp.vec3(0.)
        self.w = wp.vec3(0.)                # Camera frame basis vectors

        self.image_height = max(1, int(self.image_width / self.aspect_ratio))
        self.pixel_samples_scale = 1. / float(self.samples_per_pixel)
        focal_length = wp.length(self.lookat - self.center)
        theta = self.vfov / 180. * 3.1415926535
        h = wp.tan(theta / 2.)
        viewport_height = 2. * h * focal_length
        viewport_width = viewport_height * float(self.image_width) / float(self.image_height)
        self.w = wp.normalize(self.center - self.lookat)
        self.u = wp.normalize(wp.cross(self.vup, self.w))
        self.v = wp.normalize(wp.cross(self.w, self.u))
        viewport_u = viewport_width * self.u
        viewport_v = viewport_height * self.v
        self.pixel_delta_u = viewport_u / float(self.image_width)
        self.pixel_delta_v = viewport_v / float(self.image_height)
        viewport_upper_left = self.center - (focal_length * self.w) \
                            - viewport_u / 2. - viewport_v / 2.
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

        ########## Init Array ##########
        shape = (self.image_height, self.image_width)
        self.image = wp.zeros(shape, dtype=wp.vec3)
        self.rays = Rays(self.image_width, self.image_height, self.samples_per_pixel)
        self.get_rays()

    def get_rays(self):
        self.rays.origin = self.center
        for index in range(self.samples_per_pixel):
            wp.launch(
                kernel=compute_rays,
                dim=(self.image_height, self.image_width),
                inputs=[self.rays.origin, self.pixel00_loc, self.pixel_delta_u, self.pixel_delta_v],
                outputs=[self.rays.direction[index]]
            )
