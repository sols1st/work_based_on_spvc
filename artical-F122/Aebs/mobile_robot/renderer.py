"""Fixed external camera. Pose is encoded by appearance, never text labels."""
import numpy as np
from PIL import Image,ImageDraw
from . import core


class RobotRenderer:
    def __init__(self,size=224):self.size=size

    def render(self,state,brightness=1.):
        n=self.size
        image=Image.new('RGB',(n,n),(214,218,210));d=ImageDraw.Draw(image)
        def px(x,y):return (float(x/6*n),float((4-y)/4*n))
        for x in np.arange(0,6.01,.5):d.line([px(x,0),px(x,4)],fill=(192,198,191),width=1)
        for y in np.arange(0,4.01,.5):d.line([px(0,y),px(6,y)],fill=(192,198,191),width=1)
        d.rectangle([px(.05,3.95),px(5.95,.05)],outline=(52,64,67),width=4)
        d.ellipse([px(4.7,2.3),px(5.3,1.7)],fill=(130,196,132),outline=(30,105,50),width=2)
        ox,oy=core.OBSTACLE;r=core.OBSTACLE_RADIUS
        d.ellipse([px(ox-r,oy+r),px(ox+r,oy-r)],fill=(96,99,112),outline=(38,40,52),width=2)
        # Fixed asymmetric landmarks provide a visual map reference.
        d.rectangle([px(.3,3.8),px(.6,3.6)],fill=(218,160,48))
        d.rectangle([px(5.3,.4),px(5.6,.2)],fill=(128,76,165))
        x,y,theta=np.asarray(state)
        c,s=np.cos(theta),np.sin(theta)
        def point(forward,lateral):return px(x+c*forward-s*lateral,y+s*forward+c*lateral)
        rear=[point(-.2,-.13),point(0,-.13),point(0,.13),point(-.2,.13)]
        front=[point(0,-.13),point(.2,0),point(0,.13)]
        d.polygon(rear,fill=(36,72,202),outline=(15,30,80))
        d.polygon(front,fill=(232,51,43),outline=(110,24,20))
        if brightness!=1:
            return np.clip(np.asarray(image,dtype=float)*brightness,0,255).astype('uint8')
        return np.asarray(image)
