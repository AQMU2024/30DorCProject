import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

R_file = '../Data/Visible/subtracted.fits'
G_file = '../Data/reproj_obs/merged_im.fits'

R_fits = fits.open(R_file)
G_fits = fits.open(G_file)

wcs = WCS(R_fits[0].header)

fr_norm = 255/np.amax(R_fits[0].data)
fg_norm = 255/np.amax(G_fits[0].data)

R_data = np.array([[[pixel,0,0] for pixel in row ] for row in R_fits[0].data])*fr_norm
G_data = np.array([[[0,pixel,0] for pixel in row ] for row in G_fits[0].data[0][0]])*fg_norm

lognorm = clrs.LogNorm(vmin=1, vmax=255)

fig = plt.figure()
ax = fig.add_subplot(projection=wcs)

imG = ax.imshow(G_data, origin='lower', cmap='hot', alpha=0.5)
imR = ax.imshow(R_data, origin='lower', cmap='hot', alpha=0.5)