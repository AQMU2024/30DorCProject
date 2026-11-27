import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.convolution import Gaussian2DKernel, convolve
from regions import LineSkyRegion

def taxicab(a,b):
    return np.sum(np.abs(a - b))

def lineToPoints(data, line, n):
    x0= line.start.to_pixel(wcs)[0]
    y0= line.start.to_pixel(wcs)[1]
    
    x1 = line.end.to_pixel(wcs)[0]
    y1 = line.end.to_pixel(wcs)[1]
    
    d = np.sqrt((x1-x0)**2  + (y1-y0)**2)
    t_arr = np.linspace(0, d, n)
    
    if n > taxicab(np.array([x0,y0]), np.array([x1, y1])):
        print("\nWARNING: Number of evaluations is larger than the number of pixels along the line. \n" +
              "\t\t The number of evaluations will be decreased to the maximum possible instead \n"
              +f"\t\t (n = {abs(int(np.min([np.round(x1 - x0), np.round(y1 - y0)])))}).")
        n = abs(int(taxicab(np.array([x0,y0]), np.array([x1, y1]))))

    theta = np.atan2(y1-y0, x1-x0)
    
    points = [(int(x0 + t*np.cos(theta)), int(y0 + t*np.sin(theta))) for t in t_arr]
    values = [data[y][x] for (x,y) in points]
    return points, values


# Import and open the fits file.
file = '../Data/Visible/subtracted.fits' 
file = '../Data/reproj_obs/merged_im.fits'
hdu = fits.open(file)[0]

wcs = WCS(hdu.header).dropaxis(2).dropaxis(2)

data = hdu.data[0][0][40:330,40:330]

shape = data.shape # Find the shape of the data table.
n_points = 10 # Number of points to draw

x_arr = []
y_arr = []
x = 0
y = 0

# Generate random points along the edge of the image
for c in range(n_points):
    side = np.random.randint(4)
    
    offset = 10
    # If a certain side is picked, pick a random coordinae along that side.
    match side:
        case 0:
            x = shape[1] - offset
            y = np.random.randint(offset, shape[0] - offset)
        case 1:
            y = shape[0] - offset
            x = np.random.randint(offset, shape[1] - offset)
        case 2:
            x = offset
            y = np.random.randint(offset, shape[0] - offset)
        case 3:
            y = offset
            x = np.random.randint(offset, shape[1] - offset)
        
    x_arr.append(x)
    y_arr.append(y)

point_arr = np.array([x_arr, y_arr]).T # Point array as pixels
sc_arr = [pixel_to_skycoord(x_pix, y_pix, wcs=wcs) for (x_pix, y_pix) in point_arr]

start_end = np.array([(start, end) for start in point_arr for end in point_arr])
start_end_sc = np.array([(start, end) for start in sc_arr for end in sc_arr])

dist_start_end = [taxicab(start, end) for start in point_arr for end in point_arr]
taxicab_filter = dist_start_end > np.max(shape)

se_sc_filt = np.array(start_end_sc[taxicab_filter])
se_sc_filt = se_sc_filt

lines = [LineSkyRegion(start, end) for (start,end) in se_sc_filt][::4]
print(lines)



line_pixels = [line.to_pixel(wcs=wcs) for line in lines]

gauss_kernel = Gaussian2DKernel(2)
smoothed_data = convolve(data,gauss_kernel)

fig = plt.figure()
ax = fig.add_subplot(projection=wcs)
ax.set_facecolor('black')
ax.set_ylabel('Dec (J2000)')
ax.set_xlabel('RA (J2000)')
im = ax.imshow(smoothed_data, origin='lower', norm=clrs.LogNorm(vmin=1, vmax=8), cmap='hot')
cbar = fig.colorbar(mappable=im, label="Counts")

for line in line_pixels:
    line.plot(ax=ax, linewidth=1, color='w', alpha=0.25)
    
for line in lines:
    points, values = lineToPoints(smoothed_data, line, 100)
    n_pix = range(len(points))
    
    fig, ax = plt.subplots()
    ax.plot(n_pix, values, c='k', linewidth=1)
    



