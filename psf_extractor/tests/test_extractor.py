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
