"""Isolated PyBullet scene from upstream run_parking_lot_dino.py.

Analytic dynamics moves the robot by teleportation; PyBullet only renders.
DIRECT/TinyRenderer is an explicit portable alternative, NOT pixel-equivalent
to the upstream GUI/OpenGL renderer. No invented visible circular obstacles.
"""
from pathlib import Path
import importlib.util
import numpy as np
from .core import RENDER_OFFSET, vector


def visible_body_pixels(segmentation, body_id):
    """PyBullet body id occupies the low 24 bits; -1 is background."""
    mask = np.asarray(segmentation, dtype=np.int64)
    return int(np.count_nonzero((mask >= 0) & ((mask & ((1 << 24)-1)) == body_id)))


class ParkingRenderer:
    def __init__(self, vision_sls_root, backend='tiny'):
        import pybullet as p
        import pybullet_data
        from pybullet_utils import bullet_client
        # Loading just this standalone helper avoids pybullet_envs/__init__.py,
        # which eagerly registers unrelated Gym environments (incompatible with
        # modern Gym). This is the original Racecar implementation, not a new car.
        helper = Path(pybullet_data.__file__).resolve().parent.parent / 'pybullet_envs/bullet/racecar.py'
        spec = importlib.util.spec_from_file_location('_spvc_parking_pybullet_racecar', helper)
        racecar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(racecar)
        self.racecar_helper = helper

        if backend not in ('tiny', 'opengl', 'egl'):
            raise ValueError('backend must be tiny, opengl or egl')
        self.root, self.backend, self.p = Path(vision_sls_root).resolve(), backend, p
        plane = self.root / 'expe/parking_lot/plane_v2/plane_v2.urdf'
        if not plane.is_file():
            raise FileNotFoundError(str(plane))
        self.client = bullet_client.BulletClient(connection_mode=p.GUI if backend == 'opengl' else p.DIRECT)
        self.egl_plugin = None
        try:
            c = self.client
            if backend == 'egl':
                spec_egl = importlib.util.find_spec('eglRenderer')
                if spec_egl is None:
                    raise RuntimeError('EGL renderer plugin unavailable; not falling back to TinyRenderer')
                self.egl_plugin = c.loadPlugin(spec_egl.origin, '_eglRendererPlugin')
                if self.egl_plugin < 0:
                    raise RuntimeError('EGL plugin failed; not falling back to TinyRenderer')
            c.setTimeStep(.001)
            c.loadURDF(str(plane))
            c.setGravity(0, 0, -10)
            self.robot = racecar.Racecar(c, urdfRootPath=pybullet_data.getDataPath(),
                                        timeStep=.001).racecarUniqueId
            col = c.createCollisionShape(p.GEOM_BOX, halfExtents=[1., .75, .01])
            vis = c.createVisualShape(p.GEOM_BOX, halfExtents=[1., .75, .01],
                                     rgbaColor=[.2, .2, .2, .97])
            box = c.createMultiBody(0, col, vis, basePosition=[1.75, 0, .55],
                                    baseOrientation=p.getQuaternionFromEuler([0., 0., 0.]))
            c.changeDynamics(box, -1, lateralFriction=.7)
            for _ in range(1000):
                c.stepSimulation()
            pos, orn = c.getBasePositionAndOrientation(self.robot)
            self.height = pos[2]
            self.orientation = list(p.getEulerFromQuaternion(orn))
            self.view = c.computeViewMatrix([1.6, 0, 2.25], [1.6, 0, 0], [0, 1., 0])
            self.projection = c.computeProjectionMatrixFOV(90, 1., .02, 25.5)
            self.last_pose_error = None
        except BaseException:
            self.close()
            raise

    def render(self, world_state):
        state = vector(world_state, 4, 'world_state') + RENDER_OFFSET
        orientation = self.orientation.copy()
        orientation[2] = state[2]
        self.client.resetBasePositionAndOrientation(
            self.robot, [state[0], state[1], self.height],
            self.p.getQuaternionFromEuler(orientation))
        actual, _ = self.client.getBasePositionAndOrientation(self.robot)
        self.last_pose_error = float(np.max(np.abs(np.asarray(actual[:2])-state[:2])))
        if self.last_pose_error > 1e-7:
            raise RuntimeError('renderer pose does not match requested state')
        image = self.client.getCameraImage(
            512, 512, self.view, self.projection, shadow=True,
            renderer=self.p.ER_TINY_RENDERER if self.backend == 'tiny'
            else self.p.ER_BULLET_HARDWARE_OPENGL)
        self.last_visible_robot_pixels = visible_body_pixels(image[4], self.robot)
        return np.asarray(image[2], dtype=np.uint8).reshape(512, 512, 4)[:, :, :3].copy()

    def close(self):
        if getattr(self, 'client', None) is not None:
            if getattr(self, 'egl_plugin', None) is not None and self.egl_plugin >= 0:
                self.client.unloadPlugin(self.egl_plugin)
                self.egl_plugin = None
            self.client.disconnect()
            self.client = None
