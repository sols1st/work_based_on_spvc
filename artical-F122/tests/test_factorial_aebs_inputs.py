import numpy as np
from PIL import Image

from Aebs.experiments.factorial_aebs_inputs import pixels
from Aebs.experiments.matched_aebs_library import gray_pixels


def test_same_source_factorial_uses_identical_spvc_gray_array(tmp_path):
    y,x=np.mgrid[:640,:640]
    rgb=np.stack([x%256,y%256,(x+2*y)%256],axis=-1).astype(np.uint8)
    image=Image.fromarray(rgb)
    path=tmp_path/'frame.png'
    image.save(path)
    low_gray,low_rgb,full_gray,full_rgb=pixels(path)
    assert low_gray.shape==(32,32)
    assert low_rgb.shape==(32,32,3)
    np.testing.assert_array_equal(low_gray,gray_pixels(image))
    np.testing.assert_array_equal(low_rgb,np.asarray(image.resize((32,32),Image.Resampling.BICUBIC)))
    np.testing.assert_array_equal(full_gray,np.asarray(image.convert('L')))
    np.testing.assert_array_equal(full_rgb,rgb)
