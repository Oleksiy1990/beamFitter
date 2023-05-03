import pytest
from lmfit import Parameters
import numpy as np

import beam_fitter.beam_fitter_cli.image as image

# ******* Testing the Image class *******************************


class TestImage:
    # paths to test images to use
    img1 = {"./sample_images/sample_1_small.bmp": (1024, 1280)}
    img2 = {"./sample_images/sample_2_weak.bmp": (1024, 1280)}
    img3 = {"./sample_images/sample_3_perfect.JPG": (362, 440)}
    img4 = {"./sample_images/sample_4_elliptical.jpg": (100, 100)}
    img5 = {"./sample_images/sample_5_elliptical.jpg": (197, 292)}
    img6 = {"./sample_images/sample_6_poor_profile.bmp": (480, 640)}
    img7 = {"./sample_images/sample_7_lots_of_speckle.jpg": (1024, 1536)}
    img_unsupported = {
        "./sample_images/sample_8_unsupported_format.svg": (50, 50)}

    # ***************** Test the constructor __init__() ********************************
    @pytest.mark.parametrize("""
                             expected_shape,
                             source_test,
                             imagepath_test,
                             pixelsize_test
                             """,
                             [  # at first we test the thrown exceptions for wrong source type, format, file not found
                                 # source type is wrong, should raise UnknownImageSourceError
                                 (image.UnknownImageSourceError,
                                  "random", list(img1.keys())[0], 0.1),
                                 # filename is wrong, should raise FileNotFoundError
                                 (FileNotFoundError, "file", "wrongfile.png", 0.1),
                                 # file extension (so image type) is wrong, should raise FileNotFoundError
                                 (image.UnknownFileExtensionError, "file",
                                  list(img_unsupported.keys())[0], 0.1)
                             ] +  # we are now appeding all the correct images for tests
                             [(list(img.values())[0],
                               "file",
                                  list(img.keys())[0],
                                  0.25) for img in [img1, img2, img3, img4, img5, img6, img7]]
                             )
    def test___init__(self,
                      expected_shape,
                      source_test, imagepath_test, pixelsize_test):
        if isinstance(expected_shape, Exception):
            with pytest.raises(expected_shape):
                image.Image(source=source_test,
                            imagepath=imagepath_test,
                            pixelsize_mm=pixelsize_test
                            )
        if isinstance(expected_shape, tuple):
            loaded_image = image.Image(source=source_test,
                                       imagepath=imagepath_test,
                                       pixelsize_mm=pixelsize_test
                                       )
            # we check for three things:
            # data attribute must be a numpy array
            # data attribute must have the correct shape
            # data attribute must correctly load the give pixel size
            # offsets must be (0,0) at loading
            assert isinstance(loaded_image.data, np.ndarray)
            assert loaded_image.data.shape == expected_shape
            assert loaded_image.pixelsize_mm == pixelsize_test
            assert (loaded_image.crop_vertical_offset,
                    loaded_image.crop_horizonal_offset) == (0, 0)

    # ***************** Test Image.crop_in_pixels() method ********************************
    @pytest.mark.parametrize("""
                             expected_shape,
                             expected_crop_offsets,
                             imagepath_test,
                             vertical_low_test,
                             vertical_high_test,
                             horizontal_low_test,
                             horizontal_high_test
                             """,
                             [
                                 # 0: test that negative index fails
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  -1, 10, 1, 20),
                                 # 1: test that non-integer index fails with pixel units
                                 (list(img1.values())[0],
                                     (0, 0),
                                     list(img1.keys())[0],
                                  1, 10, 1, 20.1),
                                 # 2 : test that cropping fails when high index is below low index
                                 (list(img4.values())[0],  # image 4 is 100x100 originally, and now we take 9 rows and 99 columns
                                     (0, 0),
                                     list(img4.keys())[0],
                                  1, 10, 20, 15),
                                 # 3 : test that cropping works even when high index is out of bounds
                                 # in this case the image is not cropped on the top index side
                                 # use image 4 for this
                                 ((9, 99),  # image 4 is 100x100 originally, and now we take 9 rows and 99 columns
                                     (1, 1),
                                     list(img4.keys())[0],
                                     1, 10, 1, 20000000),
                                 # 4 : test that cropping works with nice indices
                                 # use image 4 for this
                                 ((14, 271),  # image 2 is 1024x1280 originally, and now we take 14 rows and 271 columns
                                     (1, 2),
                                     list(img2.keys())[0],
                                     1, 15, 2, 273),
                                 # 5 : test2 that cropping works with nice indices, different image
                                 ((25, 73),
                                     (10, 100),
                                     list(img5.keys())[0],
                                     10, 35, 100, 173)
                             ]
                             )
    def test_crop_in_pixels(self,
                            expected_shape,
                            expected_crop_offsets,
                            imagepath_test,
                            vertical_low_test,
                            vertical_high_test,
                            horizontal_low_test,
                            horizontal_high_test):
        # set pixelsie for these tests, make sure to use the same value in
        # parameters above
        pixelsize_test = 0.15
        # create an Image instance
        loaded_image = image.Image(source="file",
                                   imagepath=imagepath_test,
                                   pixelsize_mm=pixelsize_test
                                   )
        # do the cropping
        loaded_image.crop_in_pixels(vertical_low_test, vertical_high_test,
                                    horizontal_low_test, horizontal_high_test)
        # now check that the image has been modified appropriately
        assert isinstance(loaded_image.data, np.ndarray)
        assert loaded_image.data.shape == expected_shape
        assert loaded_image.pixelsize_mm == pixelsize_test
        assert (loaded_image.crop_vertical_offset,
                loaded_image.crop_horizonal_offset) == expected_crop_offsets

    # ***************** Test Image.crop_in_mm() method ********************************
    @pytest.mark.parametrize("""
                             expected_shape,
                             expected_crop_offsets,
                             imagepath_test,
                             vertical_low_test,
                             vertical_high_test,
                             horizontal_low_test,
                             horizontal_high_test
                             """,
                             [
                                 # 0: test that negative index on the bounds fails
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  -0.5, 10, 1, 20),
                                 # 1: test that cropping fails when high index is below low index
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  0.5, 3, 2.1, 1.1),
                                 # 2 : test that cropping works even when high index is out of bounds
                                 # in this case the image is not cropped on the horizonal top index side
                                 # use image 4 for this
                                 ((int(3.2/0.15) - int(1.7/0.15), 100 - int(1./0.15)),  # image 4 is 100x100 originally
                                     (int(1.7/0.15), int(1/0.15)),
                                     list(img4.keys())[0],
                                  1.7, 3.2, 1., 20000000),
                                 # 3 : test that cropping works with nice indices
                                 ((int(7.9/0.15)-int(5.7/0.15), int(10.5/0.15)-int(0.1/0.15)),  # image 2 is 1024x1280 originally, and now we take 14 rows and 271 columns
                                     (int(5.7/0.15), int(0.1/0.15)),
                                     list(img2.keys())[0],
                                     5.7, 7.9, 0.1, 10.5),
                                 # 4 : test2 that cropping works with nice indices, different image
                                 ((int(1/0.15)-int(0/0.15), int(2/0.15)-int(1/0.15)),
                                     (int(0/0.15), int(1/0.15)),
                                     list(img5.keys())[0],
                                     0, 1, 1, 2)
                             ]
                             )
    def test_crop_in_mm(self,
                        expected_shape,
                        expected_crop_offsets,
                        imagepath_test,
                        vertical_low_test,
                        vertical_high_test,
                        horizontal_low_test,
                        horizontal_high_test):
        # set pixelsie for these tests, make sure to use the same value in
        # parameters above
        pixelsize_test = 0.15
        # create an Image instance
        loaded_image = image.Image(source="file",
                                   imagepath=imagepath_test,
                                   pixelsize_mm=pixelsize_test
                                   )
        # do the cropping
        loaded_image.crop_in_mm(vertical_low_test, vertical_high_test,
                                horizontal_low_test, horizontal_high_test)
        # now check that the image has been modified appropriately
        assert isinstance(loaded_image.data, np.ndarray)
        assert loaded_image.data.shape == expected_shape
        assert loaded_image.pixelsize_mm == pixelsize_test
        assert (loaded_image.crop_vertical_offset,
                loaded_image.crop_horizonal_offset) == expected_crop_offsets

    # ***************** Test Image.crop_in_pixels_centered() method ********************************

    @pytest.mark.parametrize("""
                             expected_shape,
                             expected_crop_offsets,
                             imagepath_test,
                             vertical_center_test,
                             vertical_halfwidth_test,
                             horizontal_center_test,
                             horizontal_halfwidth_test
                             """,
                             [
                                 # 0: test that negative on center or span index fails
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  -1, 10, 1, 20),
                                 # 1: test2 that negative on center or span index fails
                                 (list(img1.values())[0],
                                     (0, 0),
                                     list(img1.keys())[0],
                                  1, -10, 1, 20),
                                 # 2: test that non-integer index fails with pixel units
                                 (list(img1.values())[0],
                                     (0, 0),
                                     list(img1.keys())[0],
                                  1, 10, 1.7, 20),
                                 # 3 : test that cropping works even when span is larger than image size
                                 # in this case the image is not cropped on the horizonal top side
                                 # use image 4 for this
                                 ((21, 100),
                                     (40, 0),
                                     list(img4.keys())[0],
                                     50, 10, 1, 20000000),
                                 # 4 : test that cropping works with nice indices
                                 # use image 4 for this
                                 ((31, 501),  # image 2 is 1024x1280 originally
                                     (485, 50),
                                     list(img2.keys())[0],
                                     500, 15, 300, 250)
                             ]
                             )
    def test_crop_in_pixels_centered(self,
                                     expected_shape,
                                     expected_crop_offsets,
                                     imagepath_test,
                                     vertical_center_test,
                                     vertical_halfwidth_test,
                                     horizontal_center_test,
                                     horizontal_halfwidth_test):
        # set pixelsie for these tests, make sure to use the same value in
        # parameters above
        pixelsize_test = 0.15
        # create an Image instance
        loaded_image = image.Image(source="file",
                                   imagepath=imagepath_test,
                                   pixelsize_mm=pixelsize_test
                                   )
        # do the cropping
        loaded_image.crop_in_pixels_centered(vertical_center_test, vertical_halfwidth_test,
                                             horizontal_center_test, horizontal_halfwidth_test)
        # now check that the image has been modified appropriately
        assert isinstance(loaded_image.data, np.ndarray)
        assert loaded_image.data.shape == expected_shape
        assert loaded_image.pixelsize_mm == pixelsize_test
        assert (loaded_image.crop_vertical_offset,
                loaded_image.crop_horizonal_offset) == expected_crop_offsets

    # ***************** Test Image.crop_in_mm_centered() method ********************************
    @pytest.mark.parametrize("""
                             expected_shape,
                             expected_crop_offsets,
                             imagepath_test,
                             vertical_center_test,
                             vertical_halfwidth_test,
                             horizontal_center_test,
                             horizontal_halfwidth_test
                             """,
                             [
                                 # 0: test that negative on center or span
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  -1, 10, 1, 20),
                                 # 1: test2 that negative on center or span
                                 (list(img1.values())[0],
                                  (0, 0),
                                     list(img1.keys())[0],
                                  1.2, -10.7, 1.2, 20.1),
                                 # test that if range goes outside the area, nothing gets cropped
                                 # we set pixelsize_mm to 0.15
                                 (list(img2.values())[0],  # image 2 is 1024x1280 originally
                                     (0, 0),
                                     list(img2.keys())[0],
                                  0.75, 1000, 0.5, 1000),
                                 # test that nice indices in mm also work
                                 # we set pixelsize_to 0.15
                                 ((2*int(0.7/0.15) + 1, 2*int(0.4/0.15) + 1),  # image 2 is 1024x1280 originally
                                     (int(1.3/0.15)-int(0.7/0.15),
                                      int(0.7/0.15)-int(0.4/0.15)),
                                     list(img2.keys())[0],
                                     1.3, 0.7, 0.7, 0.4),
                                 # test2 that nice indices
                                 # we set pixelsize_to 0.15
                                 ((2*int(1.3/0.15) + 1, 2*int(0.7/0.15) + 1),  # image 2 is 1024x1280 originally
                                     (int(2.8/0.15)-int(1.3/0.15),
                                      int(3.3/0.15)-int(0.7/0.15)),
                                     list(img2.keys())[0],
                                     2.8, 1.3, 3.3, 0.7)
                             ]
                             )
    def test_crop_in_mm_centered(self,
                                 expected_shape,
                                 expected_crop_offsets,
                                 imagepath_test,
                                 vertical_center_test,
                                 vertical_halfwidth_test,
                                 horizontal_center_test,
                                 horizontal_halfwidth_test):
        # set pixelsie for these tests, make sure to use the same value in
        # parameters above
        pixelsize_test = 0.15
        # create an Image instance
        loaded_image = image.Image(source="file",
                                   imagepath=imagepath_test,
                                   pixelsize_mm=pixelsize_test
                                   )
        # do the cropping
        loaded_image.crop_in_mm_centered(vertical_center_test, vertical_halfwidth_test,
                                         horizontal_center_test, horizontal_halfwidth_test)
        # now check that the image has been modified appropriately
        assert isinstance(loaded_image.data, np.ndarray)
        assert loaded_image.data.shape == expected_shape
        assert loaded_image.pixelsize_mm == pixelsize_test
        assert (loaded_image.crop_vertical_offset,
                loaded_image.crop_horizonal_offset) == expected_crop_offsets
