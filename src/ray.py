import warp as wp

class Rays:
    def __init__(self, image_width, image_height, samples_per_pixel):
        shape = (image_height, image_width)
        self.origin = wp.vec3(0.)
        self.direction = []
        for _ in range(samples_per_pixel):
            self.direction.append(wp.zeros(shape, dtype=wp.vec3))

    # def get_p(self, t: float):
    #     ''' Compute the position of ray at t '''
    #     return self.origin + self.direction * t
    
    # def get_t(self, p: wp.vec3):
    #     ''' Compute the t to reach p (if can)'''
    #     return (p[0] - self.origin[0]) / self.direction[0]
