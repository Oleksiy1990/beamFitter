1. sample_1_small.bmp
    Small and intense laser spot (small in comparison to the size of the image)
    Dimensions: 1280x1024 pixels
2. sample_2_weak.bmp
    Weak laser spot (max intensity low)
    Dimensions: 1280x1024 pixels
3. sample_3_perfect.JPG
    Essentially a perfect Gaussian profile, with size very close to picture size
    Dimensions: 440x362 pixels
4. sample_4_elliptical.jpg
    Elliptical profile, possibly for testing rotated 2D fitting
    Dimensions: 100x100 pixels
5. sample_5_elliptical.jpg
    Elliptical profile, possibly for testing rotated 2D fitting
    Dimensions: 292x197 pixels
6. sample_6_poor_profile.bmp
    Here we have a beam that is kind of Gaussian profile, 
    but it has a lot of distortion.
    Dimensions: 640x480 pixels
7. sample_7_lots_of_speckle.jpg
    Kind of a Gaussian profile, but quite strongly distorted 
    by speckle
    Dimensions: 1536x1024 pixels
8. sample_8_unsupported_format.svg
    This image should not load (unless the __init__ of Image is modified)
    and this should in general be a format that cannot be loaded 
    so that one can test the appropriate exception