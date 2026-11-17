"""Deterministic top-down/perspective road renderer for lane keeping.

Only numpy and Pillow are used, so the visual example does not require
PyBullet/CARLA.  The camera sits on the ego vehicle; the road is a single lane
with solid edge lines and a dashed centre guide.  Lighting, camera pose and a
roadside occluder are randomised by ``data.py`` so the learned observation has
a non-trivial residual distribution.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .core import E_MAX, THETA_MAX, PHI_MAX

ROAD_HALF = 1.75
GRASS = (96, 112, 78)
ASPHALT = (78, 78, 82)
EDGE = (238, 238, 232)
SKY_TOP = (120, 158, 198)
SKY_BOTTOM = (206, 219, 226)


class Camera:
    def __init__(self, eye, target, width=256, height=256, vfov_deg=55., up=(0., 0., 1.), near=.25):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.width, self.height, self.near = int(width), int(height), float(near)
        forward = np.asarray(target, dtype=np.float64) - self.eye
        forward /= np.linalg.norm(forward)
        up = np.asarray(up, dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.basis = np.stack([right, down, forward])
        self.focal = .5 * self.height / np.tan(np.deg2rad(vfov_deg) / 2)

    def to_camera(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (points - self.eye) @ self.basis.T

    def project(self, cam_points):
        cam_points = np.asarray(cam_points, dtype=np.float64)
        z = np.maximum(cam_points[:, 2], 1e-9)
        return np.stack([self.width / 2 + self.focal * cam_points[:, 0] / z,
                         self.height / 2 + self.focal * cam_points[:, 1] / z], axis=1)


class Face:
    __slots__ = ('points', 'color', 'shaded', 'layer')

    def __init__(self, points, color, shaded=True, layer=0):
        self.points = np.asarray(points, dtype=np.float64)
        self.color = tuple(color)
        self.shaded = shaded
        self.layer = layer


def _clip_near(cam_points, near):
    out, n = [], len(cam_points)
    for i in range(n):
        a, b = cam_points[i], cam_points[(i + 1) % n]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.asarray(out)


def _shade(color, normal, light=(.35, -.55, .75), ambient=.55):
    n = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return tuple(int(c) for c in color)
    n /= norm
    l = np.asarray(light, dtype=np.float64)
    l /= np.linalg.norm(l)
    k = ambient + (1 - ambient) * max(0., float(n @ l))
    return tuple(int(np.clip(c * k, 0, 255)) for c in color)


def _box_faces(center, yaw, length, width, height, color, layer=1):
    cx, cy = center
    c, s = np.cos(yaw), np.sin(yaw)
    corners = []
    for dx, dy in ((-width / 2, -length / 2), (width / 2, -length / 2),
                   (width / 2, length / 2), (-width / 2, length / 2)):
        corners.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    low = [(x, y, 0.) for x, y in corners]
    high = [(x, y, height) for x, y in corners]
    faces = [Face(low[::-1], color, layer=layer), Face(high, color, layer=layer)]
    for i in range(4):
        faces.append(Face([low[i], low[(i + 1) % 4], high[(i + 1) % 4], high[i]], color, layer=layer))
    return faces


def _centerline(kappa, s):
    if abs(kappa) < 1e-9:
        return np.array([0., s, 0.])
    psi = kappa * s
    return np.array([(1. - np.cos(psi)) / kappa, np.sin(psi) / kappa, psi])


def _offset_point(kappa, s, lateral, z=0.):
    x, y, psi = _centerline(kappa, s)
    return np.array([x + np.cos(psi) * lateral, y - np.sin(psi) * lateral, z])


def _strip(kappa, s0, s1, lat0, lat1, z, color, segments=1, layer=0, shaded=True):
    faces = []
    edges = np.linspace(s0, s1, segments + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        quad = [_offset_point(kappa, a, lat0, z), _offset_point(kappa, b, lat0, z),
                _offset_point(kappa, b, lat1, z), _offset_point(kappa, a, lat1, z)]
        faces.append(Face(quad, color, shaded=shaded, layer=layer))
    return faces


class RoadRenderer:
    def __init__(self, width=256, height=256, fov=55.):
        self.width, self.height, self.fov = int(width), int(height), float(fov)

    def render(self, state, kappa=0., lighting=1., occluder=False,
               camera_lateral=0., camera_height=1.25, blur=0., noise=0., seed=0):
        e, theta = float(state[0]), float(state[1])
        tint = lambda c: tuple(int(np.clip(v * lighting, 0, 255)) for v in c)
        faces = [Face([(-80, -80, 0), (80, -80, 0), (80, 80, 0), (-80, 80, 0)], tint(GRASS), layer=-2)]
        faces += _strip(kappa, -4., 70., -ROAD_HALF, ROAD_HALF, .002, tint(ASPHALT), 40, layer=-1)
        for lat0, lat1 in ((ROAD_HALF - .12, ROAD_HALF), (-ROAD_HALF, -ROAD_HALF + .12)):
            for face in _strip(kappa, -4., 70., lat0, lat1, .006, tint(EDGE), 40, layer=0):
                face.shaded = False
                faces.append(face)
        for s in np.arange(6., 70., 6.):
            for face in _strip(kappa, s, s + 2.2, -.07, .07, .006, tint(EDGE), layer=0):
                face.shaded = False
                faces.append(face)
        for s in np.arange(0., 70., 12.):
            for lateral in (-ROAD_HALF - 1.1, ROAD_HALF + 1.1):
                p = _offset_point(kappa, s, lateral)
                faces += _box_faces((p[0], p[1]), kappa * s, .14, .14, 1.1, tint((200, 60, 55)))
        if occluder:
            p = _offset_point(kappa, 7., -ROAD_HALF - 1.4)
            faces += _box_faces((p[0], p[1]), 0., 2.4, 1.2, 2.2, tint((70, 70, 74)))
        eye = np.array([e + camera_lateral, 0., camera_height])
        forward = np.array([np.sin(theta), np.cos(theta), -.055])
        forward /= np.linalg.norm(forward)
        camera = Camera(eye, eye + 10 * forward, self.width, self.height, self.fov)
        image = self._paint(faces, camera)
        if blur > 0:
            image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(blur)))
        if noise > 0:
            rng = np.random.default_rng(seed)
            image = np.clip(image.astype(np.float64) +
                            rng.normal(0., noise, image.shape), 0, 255).astype(np.uint8)
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        draw.polygon([(0, self.height), (self.width, self.height),
                      (self.width, self.height - 24), (0, self.height - 32)], fill=(34, 34, 37))
        return np.asarray(pil)

    def _paint(self, faces, camera):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        top, bottom = np.asarray(SKY_TOP), np.asarray(SKY_BOTTOM)
        ramp = np.linspace(0, 1, self.height)[:, None]
        img[:] = (top[None, None, :] * (1 - ramp[:, :, None]) +
                  bottom[None, None, :] * ramp[:, :, None]).astype(np.uint8)
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        prepared = []
        for face in faces:
            cam = camera.to_camera(face.points)
            if np.all(cam[:, 2] < camera.near):
                continue
            clipped = _clip_near(cam, camera.near)
            if len(clipped) < 3:
                continue
            prepared.append((face.layer, float(np.mean(clipped[:, 2])), face, clipped))
        prepared.sort(key=lambda item: (item[0], -item[1]))
        for _, _, face, clipped in prepared:
            color = face.color
            if face.shaded:
                normal = np.cross(face.points[1] - face.points[0], face.points[2] - face.points[0])
                color = _shade(color, normal)
            uv = camera.project(clipped)
            draw.polygon([(float(u), float(v)) for u, v in uv], fill=color)
        return np.asarray(image)
