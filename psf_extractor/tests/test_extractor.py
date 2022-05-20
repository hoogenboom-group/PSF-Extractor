import numpy as np

from ..extractor import load_stack, get_min_masses
from ..util import generate_image


class TestStack:

    def test_list_tif(self):
        # List of tif files from data module
        file_pattern = [
            './data/sample_zstack/sample_zstack00.tif',
            './data/sample_zstack/sample_zstack01.tif',
            './data/sample_zstack/sample_zstack02.tif',
            './data/sample_zstack/sample_zstack03.tif',
            './data/sample_zstack/sample_zstack04.tif',
            './data/sample_zstack/sample_zstack05.tif',
        ]
        # Create stack
        stack = load_stack(file_pattern)
        # Known shape
        assert stack.shape == (6, 512, 512)

    def test_directory_tif(self):
        # Path to png stack directory
        file_pattern = './data/sample_zstack/'
        # Create stack
        stack = load_stack(file_pattern)
        # Known shape
        assert stack.shape == (34, 512, 512)


class TestMassFiltering:

    def test_min_mass(self):
        # Create artificial fluorescence image
        image = generate_image(nx=300, ny=300, N_features=20, seed=37)
        min_masses = get_min_masses(image, dx=9)
        np.testing.assert_allclose(min_masses, [ 12.21489226,  23.25292776,
                                                 44.26552752,  84.26624581,
                                                 160.41377073, 305.37230648],
                                                 rtol=1e-07, atol=0)


class TestFeatures:

    def test(self):
        pass