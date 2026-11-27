import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.colors as clrs

con_file = '../Data/Visible/30DorC_673_c.fits'
Ha_file = '../Data/Visible/30DorC_Ha_c.fits'

con_fits = fits.open(con_file)
Ha_fits = fits.open(Ha_file)

sub_array = Ha_fits[0].data - con_fits[0].data
sub_array[sub_array < 0] = 0
Ha_fits[0].data += sub_array




con_fits[0].data = sub_array

con_fits.writeto("../Data/Visible/subtracted.fits", overwrite=True)

fig, ax = plt.subplots(dpi=800)

im = ax.imshow(sub_array, origin='lower', norm=clrs.LogNorm(vmin=1e1, vmax=5e3),
               cmap='hot')

ax.set_facecolor('k')
cbar = plt.colorbar(mappable=im)

