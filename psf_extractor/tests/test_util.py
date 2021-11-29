import numpy as np
from ..util import generate_image, white_noise, nonwhite_noise


class TestGenerateImage:

    def test_generate_image(self):
        # Generate image
        image = generate_image(nx=5, ny=5, N_features=50, seed=37)
        image = np.round(100*image, decimals=2).astype(int)
        np.testing.assert_array_equal(image, [[100,   0,   0,   0,   0],
                                              [  0,   0,   0,   0,   0],
                                              [  0,   0,   0,   0,   0],
                                              [  0,   0,   3,  21,   0],
                                              [  0,   0,   0,   0,   5]])
