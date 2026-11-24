"""Exact fixed-road RGB renderer with cached geometry and world-space colors.

Only state-independent face construction/shading is cached. Projection, near
clipping, polygon ordering and painting are the reference implementation.
"""
import numpy as np
from PIL import Image, ImageDraw
from Aebs.lane_keeping.renderer import RoadRenderer, Camera, Face, _shade


class StaticRoadCamera(RoadRenderer):
    def __init__(self):
        super().__init__(192,192)
        self.faces=None
        super().render([0.,0.],0.,lighting=1.,occluder=False,camera_lateral=0.,camera_height=1.25,blur=0.,noise=0.)

    def _paint(self,faces,camera):
        if self.faces is None:
            cached=[]
            for f in faces:
                color=f.color
                if f.shaded:
                    normal=np.cross(f.points[1]-f.points[0],f.points[2]-f.points[0])
                    color=_shade(color,normal)
                cached.append(Face(f.points.copy(),color,shaded=False,layer=f.layer))
            self.faces=cached
        return super()._paint(self.faces,camera)

    def frame(self,state_m_rad):
        e,theta=map(float,state_m_rad)
        eye=np.array([e,0.,1.25])
        forward=np.array([np.sin(theta),np.cos(theta),-.055])
        forward/=np.linalg.norm(forward)
        camera=Camera(eye,eye+10*forward,self.width,self.height,self.fov)
        image=self._paint(self.faces,camera)
        pil=Image.fromarray(image);draw=ImageDraw.Draw(pil)
        draw.polygon([(0,self.height),(self.width,self.height),
                      (self.width,self.height-24),(0,self.height-32)],fill=(34,34,37))
        return np.asarray(pil)
