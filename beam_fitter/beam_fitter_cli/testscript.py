import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.ndimage import gaussian_filter1d
import sys


import fitterroutines
img = "../../ExampleImages/framebefore1.bmp"
Img1 = fitterroutines.Image(source="file",imagepath=img)
Img1.pixelsize_mm = 2.54
Img1.plotimage()
ftr1 = fitterroutines.Fitter(Img1)
ftr1.convert_pixels_to_mm(True)
ftr1.fitandplot_integrated_axis("horizontal")
sys.exit(0)
def gaussian_opt(x, amp, width, h_offset, v_offset):
    return amp*np.exp(-2*np.power(x-h_offset,2.)/np.power(width,2.)) + v_offset

random_generator = default_rng()
random_numbers = random_generator.uniform(0,3,200)

xvalues = np.linspace(0, 10, 200)
gaussian_values = gaussian_opt(xvalues, 100, 1, 3, 1) + random_numbers

gaussian_smoothed = gaussian_filter1d(gaussian_values, 100)
plt.plot(xvalues , gaussian_values)
plt.plot(xvalues , gaussian_smoothed)
plt.show()


