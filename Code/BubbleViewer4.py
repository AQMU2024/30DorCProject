import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from regions import CircleSphericalSkyRegion, TextSkyRegion, RegionMeta 
from astropy.coordinates import Angle, SkyCoord, get_icrs_coordinates, offset_by
import astropy.units as units
import astropy.utils.data

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import os
from ciao_contrib.smooth import *

ev_file = "../Data/reproj_obs/merged_evt.fits" # Merged/event file
im_file = "../Data/reproj_obs/merged_im.fits" # Image file for WCS

N_ob = 5 # Number of observations used during reprojection

ev_fits = fits.open(ev_file)
im_fits = fits.open(im_file)

# Take useful data from the fits file.
obs_id = ev_fits["PRIMARY"].header["OBS_ID"]
x_vals = ev_fits[1].data["x"]
y_vals = ev_fits[1].data["y"]
energies = ev_fits[1].data["energy"]

N_bins = 1000

# Split the image into multiple sequential energy bands.
E_range = [2e3, 5e3]
splits = 2
energy_bin_edges = np.linspace(E_range[0], E_range[1], splits)
energy_bins = np.round([[energy_bin_edges[i], energy_bin_edges[i+1]] 
                        for i in range(splits-1)], 3)

colnames = ev_fits[1].columns.names
xcol = colnames.index('x') + 1
ycol = colnames.index('y') + 1

# Get the coordinate system of the fits file (Where is the WCS info for the event file?)
coordsys = WCS(im_fits[0].header)

# Plot the 2D histogram, an image of 30 Dor C.
fig = plt.figure(dpi=500)
ax = fig.add_subplot(projection=coordsys, slices=('x','y',0,0))

raw_im = ax.imshow(im_fits[0].data[0][0], origin='lower', cmap='hot',                  
                     norm=clrs.LogNorm(vmin=1), interpolation='auto')

# Use sherpa to smooth the image (then display using imshow).
sm_fig = plt.figure(dpi=500)
sm_ax = sm_fig.add_subplot(projection=coordsys, slices=('x','y',0,0))

smooth_arr = gsmooth(im_fits[0].data[0][0], sigma=1)
sm_im = sm_ax.imshow(smooth_arr, origin='lower', cmap='hot',
                     norm=clrs.LogNorm(vmin=1))

# ===================
#   DEFINE REGIONS OF INTEREST
# ===================
"""
centre = get_icrs_coordinates(name='SN 1987A').Conf(remote_timeout=40)
radius = Angle(20, unit=units.arcsec)
region = CircleSphericalSkyRegion(centre, radius)
print(offset_by(centre.ra, centre.dec, 3*np.pi/4, Angle("0d0m10s", unit=units.arcsec)))
boundary = region.to_pixel(wcs=coordsys.sub([1,2]))
"""

for axis in [ax, sm_ax]:
    axis.grid(linewidth=0.5, alpha=0.15, linestyle='dashed')
    
    axis.set_facecolor('k')

    axis_ra = axis.coords['ra']
    axis_dec = axis.coords['dec']

    axis_ra.set_axislabel("Right Ascension (J2000)", minpad=1.5)
    axis_dec.set_axislabel("Declination (J2000)", minpad=2)

    axis_dec.set_ticks(direction='in', color='w')
    axis_ra.set_ticks(direction='in', color='w', width=1)

    axis_dec.set_ticklabel(rotation=90, size=7, pad=-15, color='w')
    axis_ra.set_ticklabel(rotation=0, size=7, pad=-15, color='w')

    axis_ra.display_minor_ticks(True)
    axis_dec.display_minor_ticks(True)

    cbar = fig.colorbar(raw_im, ax=axis)
    cbar.ax.set_ylabel("Counts")
    
    #boundary.plot(ax=axis, color='lime', alpha=0.5)


plt.show()
