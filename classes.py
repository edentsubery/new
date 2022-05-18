from dataclasses import dataclass

import numpy as np

from utils import norm2

TANGENT_TOLERANCE = 0.01

@dataclass
class Ray:
    origin:    np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        self.direction /= norm2(self.direction)

    def project(self, point):
        return np.dot(self.direction, point - self.origin)

    def get_point(self, t):
        return self.origin + t * self.direction

@dataclass
class Camera:
    position:     np.ndarray
    look_at:      np.ndarray
    up:           np.ndarray
    screen_dist:  float
    screen_width: float
    screen_height: float


@dataclass
class Set:
    background_rgb:   np.ndarray
    root_shadow_rays: int
    max_recursions:   int


@dataclass
class Material:
    diffuse_rgb:  np.ndarray
    specular_rgb: np.ndarray
    reflect_rgb:  np.ndarray
    phong:        float
    transp:       float


@dataclass
class Light:
    position:        np.ndarray
    rgb:             np.ndarray
    specular_intensity: float
    shadow_intensity:   float
    radius:          float


@dataclass
class Shape:
    material: int

    def find_intersection(self, ray: Ray):
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')

    def normal_at_point(self, point: np.array):
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')


@dataclass
class Sphere(Shape):
    center: np.ndarray
    radius: float

    def find_intersection(self, ray: Ray):
        c = self.center
        r = self.radius
        ob_size = ray.project(c)
        if ob_size <= 0:
            return False

        oc = c - ray.origin
        oc_size = norm2(oc)
        cb_size = np.sqrt(oc_size ** 2 - ob_size ** 2)
        if np.isclose(cb_size, r, rtol=TANGENT_TOLERANCE):
            return ray.get_point(ob_size)

        if cb_size > r:
            return False

        t_diff_abs = np.sqrt(r ** 2 - cb_size ** 2)
        p1 = ray.get_point(ob_size - t_diff_abs)
        p2 = ray.get_point(ob_size + t_diff_abs)
        if np.linalg.norm(ray.origin-p1) < np.linalg.norm(ray.origin-p2):
            return p1
        else:
            return p2

    def normal_at_point(self, point: np.array):
        normal_vec = point-self.center
        normal_vec = normal_vec/np.linalg.norm(normal_vec)
        return normal_vec

@dataclass
class Plane(Shape):
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        self.normal /= norm2(self.normal)

    def find_intersection(self, ray: Ray):

        n_dot_d = np.dot(self.normal, ray.direction)
        if np.isclose(n_dot_d, 0, rtol=0.01):
            return False
        n_dot_o = np.dot(self.normal, ray.origin)
        t = (self.offset - n_dot_o) / n_dot_d
        point = ray.get_point(t)
        return point

    def normal_at_point(self, point: np.array):
        return self.normal

@dataclass
class Box(Shape):
    center: np.ndarray
    length: float
    min: np.ndarray
    max: np.ndarray

    def find_intersection(self, ray: Ray):
        half_length = self.length / 2
        box_min = self.center - half_length
        box_max = self.center + half_length
        t_min = np.divide(box_min - ray.origin, ray.direction)
        t_max = np.divide(box_max - ray.origin, ray.direction)

        if t_min[0] > min(t_max[1], t_max[2]) | t_min[1] > min(t_max[0], t_max[2]) | t_min[2] > min(t_max[0], t_max[1]):
            return False
        t = t_min.max()
        return ray.get_point(t)

    def normal_at_point(self, point:np.array):
        normal = np.zeros(3)
        if point[0] == self.min[0]:
            normal += [-1, 0, 0]
        elif point[0] == self.max[0]:
            normal += [1, 0, 0]
        if point[1] == self.min[1]:
            normal += [0, -1, 0]
        elif point[1] == self.max[1]:
            normal += [0, 1, 0]
        if point[2] == self.min[2]:
            normal += [0, 0, -1]
        elif point[2] == self.max[2]:
            normal += [0, 0, 1]
        return normal / np.linalg.norm(normal)
