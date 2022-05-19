import argparse
import numpy as np
from PIL import Image
import math
import sys
from objects import Camera, Set, Material, Light, Sphere, Plane, Box, Ray

def main(argv):
    scene_txt = argv[1]
    img_name = argv[2]
    width = int(argv[3])
    height = int(argv[4])
    camera, settings, mtl, lgt, shapes = read_txt(scene_txt, width, height)
    img = RayTracing(height, width, camera, settings, mtl, lgt, shapes)
    save_img(img, img_name)

def read_txt(scene_txt, width, height):
    with open(scene_txt, 'r') as file:
        mtl = []
        lgt = []
        shapes = []
        for line in file.readlines():
            line = line.strip()
            if (len(line) == 0 or line.startswith('#')):
                continue
            instruction = line[:3]
            params = [float(v) for v in line[3:].split()]
            if (instruction == 'cam'):
                camera = Camera(position=np.array(params[0:3]), look_at=np.array(params[3:6]), up_vector=np.array(params[6:9]), screen_distance=params[9], screen_width=params[10], screen_height=params[10]/width*height)
            elif (instruction == 'set'):
                settings = Set(background_color=np.array(params[0:3]), shadow_rays=int(params[3]), max_depth=int(params[4]))
            elif (instruction == 'mtl'):
                mtl.append(Material(diffuse_color=np.array(params[0:3]), specular_color=np.array(params[3:6]), reflection_color=np.array(params[6:9]), phong=params[9], transparency=params[10]))
            elif (instruction == 'sph'):
                shapes.append(Sphere(center=np.array(params[0:3]), radius=params[3], material=int(params[4])))
            elif (instruction == 'pln'):
                shapes.append(Plane(normal=np.array(params[0:3]), offset=params[3], material=int(params[4])))
            elif (instruction == 'box'):
                shapes.append(Box(center=np.array(params[0:3]), length=params[3], material=int(params[4]),
                                  min= np.array(params[0] - (params[3] / 2), params[1] - (params[3]/ 2), params[2] - (params[3] / 2)),
                                  max=np.array(params[0] + (params[3] / 2), params[1] + (params[3]/ 2), params[2] + (params[3] / 2))))
            elif (instruction == 'lgt'):
                lgt.append(Light(position=np.array(params[0:3]), color=np.array(params[3:6]), specular_intensity=params[6], shadow_intensity=params[7], radius=params[8]))
    return camera, settings, mtl, lgt, shapes


def RayTracing(height, width, camera, settings, mtl, lgt, shapes):
    img = np.zeros([height, width, 3], dtype=np.float32)
    for i in range(width):
        for j in range(height):
            img[height - 1 - j][i], _ = algorithm_for_each_pixel(camera, i, j, width, height, img, settings, mtl, lgt, shapes)
    return img

def algorithm_for_each_pixel(camera, i, j, width, height, img, settings, mtl, lgt, shapes):
    ray = shoot_ray(camera, (i / width - 0.5) * camera.screen_width, (j / height - 0.5) * camera.screen_height)
    intersection_point, intersected_shape, _ = nearest_intersection(ray, shapes)
    return output_color(intersection_point, intersected_shape, ray, settings, mtl, lgt, shapes)

def shoot_ray(camera, width, height):
    towards = vector_normal(camera.look_at - camera.position)
    fix = vector_normal(camera.up_vector - np.dot(camera.up_vector,towards)/np.dot(towards,towards)*towards)
    width_direction = vector_normal(np.cross(fix, towards))
    pixel_on_screen = camera.position + towards*camera.screen_distance + fix*height + width_direction*width
    direction = pixel_on_screen - camera.position
    return Ray(origin=camera.position, direction=direction)


def nearest_intersection(ray, shapes):
    nearest = None
    shape_tmp = None
    index_tmp = None
    index = 0
    for shape in shapes:
        current_intersection = shape.find_intersection(ray)
        if (current_intersection is not False):
            if np.dot(current_intersection-ray.origin, ray.direction) < 0:
                continue
            if nearest is None:
                nearest = current_intersection
                shape_tmp = shape
                index_tmp = index
            else:
                if (vector_len(ray.origin-nearest) > vector_len(ray.origin-current_intersection)):
                    nearest = current_intersection
                    shape_tmp = shape
                    index_tmp = index
        index += 1
    return nearest, shape_tmp, index_tmp

def calculate_light(intersection_point, light, shapes, intersected_shape, current_material, camera_ray, shapes_list, settings, mtl, lgt):
    light_direction = vector_normal(intersection_point - light.position)
    light_ray = Ray(origin=light.position, direction=light_direction)
    light_intersection_point, light_intersected_shape, shape_index = nearest_intersection(light_ray, shapes)
    if light_intersection_point is None:
        return np.array([0.0, 0.0, 0.0])
    if not np.all(np.isclose(light_intersection_point, intersection_point)):
        return np.array([0.0, 0.0, 0.0])
    diffuse_color, specular_color = calc_diffuse_and_specular(camera_ray, current_material, intersected_shape, light, light_direction, light_intersection_point)
    light_intensity = (1 - light.shadow_intensity) * 1 + light.shadow_intensity * precentage_hits(light_ray, settings.shadow_rays, light.radius, intersection_point, shapes)
    back_color = calc_back_color(camera_ray, current_material, lgt, light, mtl, settings, shape_index, shapes, shapes_list)
    return (light.color * light_intensity * (diffuse_color + specular_color) * (1 - current_material.transparency) + current_material.transparency * back_color)


def calc_back_color(camera_ray, current_material, lgt, light, mtl, settings, shape_index, shapes, shapes_list):
    if current_material.transparency > 0:
        shapes_list = construct_shape_list(shape_index, shapes, shapes_list)
        next_intersection_point, next_intersected_shape, _ = nearest_intersection(camera_ray, shapes_list)
        back_color, hit_background = output_color(next_intersection_point, next_intersected_shape, camera_ray, settings, mtl, lgt, shapes_list)
        if not hit_background:
            back_color = back_color * light.color
    else:
        back_color = np.array([0.0, 0.0, 0.0])
    return back_color


def construct_shape_list(shape_index, shapes, shapes_list):
    if shapes_list is None:
        if len(shapes) == 0:
            shapes_list = shapes
        else:
            shapes_list = shapes.copy()
            del shapes_list[shape_index]
    return shapes_list


def calc_diffuse_and_specular(camera_ray, current_material, intersected_shape, light, light_direction, light_intersection_point):
    surface_normal = intersected_shape.normal_at_point(light_intersection_point)
    diffuse_color = current_material.diffuse_color * np.abs(np.dot(surface_normal, -light_direction))
    reflect_direction = light_direction - 2 * np.dot(light_direction, surface_normal) * surface_normal
    specular_color = current_material.specular_color * np.power(np.abs(np.dot(reflect_direction, -camera_ray.direction)), current_material.phong) * light.specular_intensity
    return diffuse_color, specular_color


def output_color(intersection_point, intersected_shape, camera_ray, settings, mtl, lgt, shapes):
    if intersection_point is None:
        return settings.background_color, True
    current_material = mtl[intersected_shape.material-1]
    color_out = np.array([0.0, 0.0, 0.0])
    shapes_list = None
    for light in lgt:
        color_out += calculate_light(intersection_point, light, shapes, intersected_shape, current_material, camera_ray, shapes_list, settings, mtl, lgt)
    color_out[color_out > 1] = 1
    return color_out, False


def precentage_hits(light_ray, rays_number, radius, intersection_point, shapes):
    x, y = get_unit_vectors(light_ray)
    hits = 0
    range = radius / rays_number
    screen = [-radius/2+range/2, -radius/2+range/2, radius/2-range/2, radius/2-range/2]
    for i in np.linspace(screen[0], screen[2], rays_number):
        for j in np.linspace(screen[1], screen[3], rays_number):
            vector_1 = i + np.random.uniform() * range - range / 2
            vector_2 = j + np.random.uniform() * range - range / 2
            center = light_ray.origin + x * vector_1 + y * vector_2
            light_direction = vector_normal(intersection_point - center)
            light_ray = Ray(origin=center, direction=light_direction)
            light_intersection_point, light_intersected_shape, _ = nearest_intersection(light_ray, shapes)
            if light_intersection_point is None or not np.all(np.isclose(light_intersection_point, intersection_point)):
                continue
            hits += 1
    return hits/(math.pow(rays_number, 2))


def get_unit_vectors(light_ray):
    x = np.array([0, 0, 1])
    if np.all(np.isclose(x, light_ray.direction)):
        x = np.array([1, 0, 0])
    x = vector_normal(x - np.dot(x, light_ray.direction) / np.dot(light_ray.direction, light_ray.direction) * light_ray.direction)
    y = vector_normal(np.cross(x, light_ray.direction))
    return x, y


def vector_normal(vector):
    return vector / np.linalg.norm(vector)


def vector_len(vector):
    return np.linalg.norm(vector)


def save_img(img, address):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(address)

if __name__ == '__main__':
    main(sys.argv)
