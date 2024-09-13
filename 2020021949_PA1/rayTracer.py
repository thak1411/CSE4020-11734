#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import math
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 

class RayTracer:
    def __init__(self, camera, surfaces):
        '''
        Initialize RayTracer
        '''
        self.camera = camera
        self.surfaces = surfaces

    def trace(self, ray, tmin, tmax):
        '''
        Trace ray from camera to surfaces
        '''
        tbest = tmax
        tcolor = Color(0, 0, 0)
        tsurface = None
        tidx = -1
        for idx, surf in enumerate(self.surfaces):
            t = surf.intersect(ray, tmin, tbest)
            if t != -1:
                tbest = t
                tsurface = surf
                tidx = idx
        
        if tsurface is not None:
            tcolor = tsurface.get_color(ray, self.camera.get_ray_point(ray, tbest), self.surfaces, tidx)
        return tcolor
    
    def trace_all(self):
        '''
        Trace ray from camera to all pixels

        return image vector
        '''
        pw = self.camera.get_pw()
        ph = self.camera.get_ph()
        channels = 3

        img = np.zeros((ph, pw, channels), dtype = np.uint8)
        img[:,:]=0

        for h in range(ph):
            for w in range(pw):
                ray = self.camera.get_ray(h, w)
                # plane 앞에 물체가 있으면 무시하도록 tmin 설정
                c = self.trace(ray, 0, float('inf'))
                img[ph - h - 1][w] = c.toUINT8()

        rawimg = Image.fromarray(img, 'RGB')
        rawimg.save(sys.argv[1]+'.png')
    
class Shader:
    def __init__(self, lights, diffuse_color, specular_color = None, exponent = None):
        '''
        Initialize Shader
        '''
        self.lights = lights
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.exponent = exponent

    def shade(self, ray, point, normal, surfaces, surfaces_skip_index):
        '''
        Shade point with normal
        '''
        raise NotImplementedError

class Lambertian(Shader):
    def __init__(self, lights, diffuse_color):
        '''
        Initialize Lambertian Shader
        '''
        super().__init__(lights, diffuse_color)

    def shade(self, ray, point, normal, surfaces, surfaces_skip_index):
        '''
        Shade point with normal
        '''
        color = Color(0, 0, 0)

        for light in self.lights:
            position, intensity = light
            l = position - point
            l = l / math.sqrt(sum(l * l))

            v = sum(l * normal)
            if v <= 0: continue

            flag = True
            for idx, surf in enumerate(surfaces):
                if surfaces_skip_index == idx: continue
                t = surf.intersect((point, l), 0, float('inf'))
                if t != -1:
                    flag = False
                    break
            if flag: color.color += intensity * self.diffuse_color.color * v

        color.gammaCorrect(2.2)
        return color

class Phong(Shader):
    def __init__(self, lights, diffuse_color, specular_color, exponent):
        '''
        Initialize Phong Shader
        '''
        super().__init__(lights, diffuse_color, specular_color, exponent)

    def shade(self, ray, point, normal, surfaces, surfaces_skip_index):
        '''
        Shade point with normal
        '''
        color = Color(0, 0, 0)

        for light in self.lights:
            position, intensity = light
            l = position - point
            l = l / math.sqrt(sum(l * l))

            pmray = ray[0] - ray[1] - point
            pmray = pmray / math.sqrt(sum(pmray * pmray))

            h = l + pmray
            h = h / math.sqrt(sum(h * h))

            v1 = max(0, sum(l * normal))
            v2 = np.power(max(0, sum(h * normal)), self.exponent)

            flag = True
            for idx, surf in enumerate(surfaces):
                if surfaces_skip_index == idx: continue
                t = surf.intersect((point, l), 0, float('inf'))
                if t != -1:
                    flag = False
                    break
            if flag:
                color.color += intensity * self.diffuse_color.color * v1
                color.color += intensity * self.specular_color.color * v2

        color.gammaCorrect(2.2)
        return color

class Surface:
    def __init__(self, shader):
        '''
        Initialize Surface
        '''
        self.shader = shader

    def intersect(self, ray, tmin, tmax):
        '''
        Intersect ray with surface
        '''
        raise NotImplementedError

    def get_normal(self, point):
        '''
        Get normal vector of surface
        '''
        raise NotImplementedError
    
    def get_color(self, ray, point, surfaces, surfaces_skip_index):
        '''
        Get color of surface
        '''
        return self.shader.shade(ray, point, self.get_normal(point), surfaces, surfaces_skip_index)

class Sphere(Surface):
    def __init__(self, shader, center, r):
        '''
        Initialize Sphere
        '''
        super().__init__(shader)
        self.center = center
        self.r = r

    def intersect(self, ray, tmin, tmax):
        '''
        Intersect ray with sphere

        return t if intersected, else -1
        t: distance from view point to sphere
        '''
        p, d = ray
        p = p - self.center

        # lv +- sqrt(rv)
        lv = -sum(d * p)
        rv = sum(d * p) ** 2 - sum(p * p) + self.r ** 2

        if rv < 0: return -1

        t1 = lv + math.sqrt(rv)
        t2 = lv - math.sqrt(rv)

        result_t = -1
        if tmin <= t1 <= tmax: result_t = t1
        if tmin <= t2 <= tmax:
            if result_t == -1: result_t = t2
            elif result_t > t2: result_t = t2
        return result_t
    
    def get_normal(self, point):
        '''
        Get normal vector of sphere
        '''
        return (point - self.center) / self.r
    
class Box(Surface):
    def __init__(self, shader, min_point, max_point):
        '''
        Initialize Box
        '''
        super().__init__(shader)
        self.min_point = min_point
        self.max_point = max_point

    def intersect(self, ray, tmin, tmax):
        '''
        Intersect ray with box

        return t if intersected, else -1
        t: distance from view point to box
        '''
        p, d = ray

        def calc_delta(s, e, d):
            return (e - s) / d
        
        tx_min = min(calc_delta(p[0], self.min_point[0], d[0]), calc_delta(p[0], self.max_point[0], d[0]))
        tx_max = max(calc_delta(p[0], self.min_point[0], d[0]), calc_delta(p[0], self.max_point[0], d[0]))
        ty_min = min(calc_delta(p[1], self.min_point[1], d[1]), calc_delta(p[1], self.max_point[1], d[1]))
        ty_max = max(calc_delta(p[1], self.min_point[1], d[1]), calc_delta(p[1], self.max_point[1], d[1]))
        tz_min = min(calc_delta(p[2], self.min_point[2], d[2]), calc_delta(p[2], self.max_point[2], d[2]))
        tz_max = max(calc_delta(p[2], self.min_point[2], d[2]), calc_delta(p[2], self.max_point[2], d[2]))

        if tmin <= max(tx_min, ty_min, tz_min) <= min(tx_max, ty_max, tz_max):
            if max(tx_min, ty_min, tz_min) <= tmax: return max(tx_min, ty_min, tz_min)
        return -1
    
    def get_normal(self, point):
        '''
        Get normal vector of box
        '''
        my_point = point.copy()
        my_point -= self.min_point

        def float_eq(a, b):
            return abs(a - b) < 1e-4

        if float_eq(my_point[0], 0): return np.array([-1, 0, 0]).astype(np.float64)
        if float_eq(my_point[1], 0): return np.array([0, -1, 0]).astype(np.float64)
        if float_eq(my_point[2], 0): return np.array([0, 0, -1]).astype(np.float64)
        if float_eq(my_point[0], self.max_point[0] - self.min_point[0]): return np.array([1, 0, 0]).astype(np.float64)
        if float_eq(my_point[1], self.max_point[1] - self.min_point[1]): return np.array([0, 1, 0]).astype(np.float64)
        if float_eq(my_point[2], self.max_point[2] - self.min_point[2]): return np.array([0, 0, 1]).astype(np.float64)
        return np.array([0, 0, 0]).astype(np.float64)

class Camera:
    def __init__(self, point, dir, proj_normal, up, dist, width, height, pw, ph):
        '''
        Initialize Camera

        Pre computation of camera coordinate system
        u: x axis vector (unit vector)
        v: y axis vector (unit vector)
        s: starting point (0.5, 0.5)
        '''
        self.point = point
        self.dir = dir
        self.dir = self.dir / math.sqrt(sum(self.dir * self.dir))
        self.proj_normal = proj_normal
        self.proj_normal = self.proj_normal / math.sqrt(sum(self.proj_normal * self.proj_normal))
        self.up = up
        self.dist = dist
        self.width = width
        self.height = height
        self.pw = pw
        self.ph = ph

        # pre computation
        self.u = np.cross(self.dir, self.up)
        self.u = self.u / math.sqrt(sum(self.u * self.u))
        self.v = np.cross(self.u, self.dir)
        self.v = self.v / math.sqrt(sum(self.v * self.v))
        self.s = self.dir * self.dist - self.u * (self.width / pw) * (pw / 2 - .5) - self.v * (self.height / ph) * (ph / 2 - .5)

    def get_ray(self, h, w):
        '''
        Get ray from camera to (x, y) pixel
        '''
        p = self.point
        d = self.s + self.u * (self.width / self.pw) * w + self.v * (self.height / self.ph) * h
        d = d / math.sqrt(sum(d * d))

        return (p, d)
    
    def get_ray_point(self, ray, t):
        '''
        Get point of ray at t distance
        '''
        p, d = ray
        return p + d * t
    
    def get_pw(self):
        '''
        Get pw
        '''
        return self.pw
    
    def get_ph(self):
        '''
        Get ph
        '''
        return self.ph
    

class Parser:
    def __init__(self, filename):
        '''
        initialize parser
        '''
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()

    def recursive_find(self, path):
        '''
        find xml element recursively
        '''
        spt = path.split('.')
        node = self.root
        for word in spt:
            node = node.find(word)
            if node is None:
                return None
        return node
    
    def recursive_find_all(self, path):
        '''
        find all xml elements recursively
        '''
        spt = path.split('.')
        node = self.root
        for word in spt[:-1]:
            node = node.find(word)
            if node is None:
                return None
        return node.findall(spt[-1])
    
    def recursive_find_parse(self, path, default_value):
        '''
        find xml element recursively and parse it to string
        '''
        node = self.recursive_find(path)
        if node is None:
            return default_value
        return node.text
    
    def parse(self, node, default_value):
        '''
        parse xml element to string
        '''
        if node is None:
            return default_value
        return node.text

class Color:
    def __init__(self, R, G, B):
        self.color=np.array([R,G,B]).astype(np.float64)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

def initialize():
    '''
    parsing xml file and initialize RayTracer

    Return initialized RayTracer
    '''
    parser = Parser(sys.argv[1])


    # parsing camera's parameters
    view_point = np.array(list(map(float, parser.recursive_find_parse('camera.viewPoint', '0 0 0').split()))).astype(np.float64)
    view_dir = np.array(list(map(float, parser.recursive_find_parse('camera.viewDir', '0 0 -1').split()))).astype(np.float64)
    proj_normal = np.array(list(map(float, parser.recursive_find_parse('camera.projNormal', '0 0 0').split()))).astype(np.float64)
    view_up = np.array(list(map(float, parser.recursive_find_parse('camera.viewUp', '0 1 0').split()))).astype(np.float64)
    proj_distance = float(parser.recursive_find_parse('camera.projDistance', '1.0'))
    view_width = float(parser.recursive_find_parse('camera.viewWidth', '1.0'))
    view_height = float(parser.recursive_find_parse('camera.viewHeight', '1.0'))
    pw, ph = map(int, parser.recursive_find_parse('image', '512 512').split())

    camera = Camera(view_point, view_dir, proj_normal, view_up, proj_distance, view_width, view_height, pw, ph)
    
    # parsing lights
    lights = []
    node = parser.recursive_find_all('light')
    for n in node:
        position = np.array(list(map(float, parser.parse(n.find('position'), '0 0 0').split()))).astype(np.float64)
        intensity = np.array(list(map(float, parser.parse(n.find('intensity'), '1 1 1').split()))).astype(np.float64)
        lights.append((position, intensity))


    # parsing shaders 
    shaders = {}
    node = parser.recursive_find_all('shader')
    for n in node:
        name = n.get('name', None)
        stype = n.get('type', None)

        if name is None or stype is None: continue

        if stype.lower() == 'lambertian':
            diffuse_color = np.array(list(map(float, parser.parse(n.find('diffuseColor'), '1 1 1').split()))).astype(np.float64)
            shaders[name] = Lambertian(lights, Color(diffuse_color[0], diffuse_color[1], diffuse_color[2]))

        if stype.lower() == 'phong':
            diffuse_color = np.array(list(map(float, parser.parse(n.find('diffuseColor'), '1 1 1').split()))).astype(np.float64)
            specular_color = np.array(list(map(float, parser.parse(n.find('specularColor'), '1 1 1').split()))).astype(np.float64)
            exponent = float(parser.parse(n.find('exponent'), '50'))
            shaders[name] = Phong(
                lights,
                Color(diffuse_color[0], diffuse_color[1], diffuse_color[2]),
                Color(specular_color[0], specular_color[1], specular_color[2]),
                exponent
            )

    # parsing surfaces
    surfaces = []
    node = parser.recursive_find_all('surface')
    for n in node:
        ntype = n.get('type', None)

        if ntype is not None and ntype.lower() == 'sphere':
            shad = Lambertian(lights, Color(1, 1, 1))

            shader = n.find('shader')
            if shader is not None:
                stype = shader.get('ref', None)
                if stype is not None and stype in shaders:
                    shad = shaders[stype]

            center = np.array(list(map(float, parser.parse(n.find('center'), '0 0 0').split()))).astype(np.float64)
            r = float(parser.parse(n.find('radius'), '1.0'))
            surfaces.append(Sphere(shad, center, r))
        elif ntype is not None and ntype.lower() == 'box':
            shad = Lambertian(lights, Color(1, 1, 1))

            shader = n.find('shader')
            if shader is not None:
                stype = shader.get('ref', None)
                if stype is not None and stype in shaders:
                    shad = shaders[stype]

            min_point = np.array(list(map(float, parser.parse(n.find('minPt'), '0 0 0').split()))).astype(np.float64)
            max_point = np.array(list(map(float, parser.parse(n.find('maxPt'), '0 0 0').split()))).astype(np.float64)
            surfaces.append(Box(shad, min_point, max_point))

    return RayTracer(camera, surfaces)

def main():
    '''
    Console application main function
    '''

    # initialize RayTracer
    ray_tracer = initialize()

    # ray tracing & save result image
    ray_tracer.trace_all()
    
if __name__ == '__main__': main()