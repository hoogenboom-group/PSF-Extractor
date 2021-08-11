from ..extractor import get_stack


class TestStack:

    def test_list_png(self):
        # List of png files from data module
        stem = '20201130_162150_collar_0.146_yaw_1.62deg_cyls_corr_back_foc_325.3um_15deg'
        file_pattern = [
            f'./data/sample_zstack_png_sequence/{stem}0000.png',
            f'./data/sample_zstack_png_sequence/{stem}0001.png',
            f'./data/sample_zstack_png_sequence/{stem}0002.png',
            f'./data/sample_zstack_png_sequence/{stem}0003.png',
            f'./data/sample_zstack_png_sequence/{stem}0004.png',
            f'./data/sample_zstack_png_sequence/{stem}0005.png',
        ]
        # Create stack
        stack = get_stack(file_pattern)
        # Known shape
        assert stack.shape == (6, 2048, 2448)

    def test_list_tif(self):
        # List of tif files from data module
        file_pattern = [
            f'./data/sample_zstack_tif_sequence/0.png',
            f'./data/sample_zstack_tif_sequence/1.png',
            f'./data/sample_zstack_tif_sequence/2.png',
            f'./data/sample_zstack_tif_sequence/3.png',
            f'./data/sample_zstack_tif_sequence/4.png',
            f'./data/sample_zstack_tif_sequence/5.png',
        ]
        # Create stack
        stack = get_stack(file_pattern)
        # Known shape
        assert stack.shape == (6, 2048, 2448)

    def test_directory_png(self):
        # Path to png stack directory
        file_pattern = './data/sample_zstack_png_sequence/'
        # Create stack
        stack = get_stack(file_pattern)
        # Known shape
        assert stack.shape == (32, 2048, 2448)

    def test_directory_tif(self):
        # Path to png stack directory
        file_pattern = './data/sample_zstack_tif_sequence/'
        # Create stack
        stack = get_stack(file_pattern)
        # Known shape
        assert stack.shape == (32, 2048, 2448)

    def test_tif_stack(self):
        # Path to tif stack in data module
        file_pattern = './data/sample_zstack_single_tif/20201130_162150_collar' +\
                       '_0.146_yaw_1.62deg_cyls_corr_back_foc_325.3um_15deg.tif'
        # Create stack
        stack = get_stack(file_pattern)
        # Known shape
        assert stack.shape == (23, 1749, 2034)


class TestFeatures:

    def test(self):
        pass