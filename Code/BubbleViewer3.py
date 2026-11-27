import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import os
from ciao_contrib.smooth import *

def aspect_func(x):
    a = np.pi/180
    return a*x**2 + (a**3)*(x**4)/6 + (a**5)*(x**6)/120
def aspect_func_inv(x):
    return (180/np.pi)*np.asin(x)

ev_file = "../Data/reproj_obs/merged_evt.fits" # Merged/event file
im_file = "../Data/reproj_obs/merged.fov" # Image file for WCS



N_ob = 5 # Number of observations used during reprojection

ev_fits = fits.open(ev_file)
im_fits = fits.open(im_file)

# Take useful data from the fits file.
obs_id = ev_fits["PRIMARY"].header["OBS_ID"]
x_vals = ev_fits[1].data["x"]
y_vals = ev_fits[1].data["y"]
energies = ev_fits[1].data["energy"]

# Split the image into multiple sequential energy bands.
E_range = [1e3, 3e3]
splits = 2
energy_bin_edges = np.linspace(E_range[0], E_range[1], splits)
energy_bins = np.round([[energy_bin_edges[i], energy_bin_edges[i+1]] 
                        for i in range(splits-1)], 3)

# Get the coordinate system of the fits file.
coordsys = WCS(header=im_fits[0].header)
#Convert pixel values to sky-coordinates.
skycoords = coordsys.pixel_to_world(x_vals, y_vals)


# Save the sky-coordinates as separate RA and DEC arrays.
ra_vals = np.array([coord[0] for coord in skycoords.to_table().as_array()])
dec_vals = np.array([coord[1] for coord in skycoords.to_table().as_array()])

# Extract the grade of each event.
grades = ev_fits[1].data["grade"]

for Ebin in energy_bins:
    # Only accept events with a grade below or equal to 4.
    graded_x = x_vals[grades <= 4]
    graded_y = y_vals[grades <= 4]
    graded_ra = ra_vals[grades <= 4]
    graded_dec = dec_vals[grades <= 4]
    graded_energies = energies[grades <= 4]
    
    # Partition events into energy bins.
    x_valEcut = graded_x[(graded_energies >= Ebin[0]) & (graded_energies<= Ebin[1])]
    y_valEcut = graded_y[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    ra_valEcut = graded_ra[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    dec_valEcut = graded_dec[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    
    # Plot the 2D histogram, an image of 30 Dor C.
    fig = plt.figure()
    ax = fig.add_subplot(projection=coordsys)
    
    print(coordsys.array_shape)
    
    im_hist = np.histogram2d(x_valEcut, y_valEcut, bins=coordsys.array_shape)
    im_arr = im_hist[0]
    
    raw_im = ax.imshow(im_arr.T, origin='lower', cmap='hot',
                         norm=clrs.LogNorm(vmin=1), interpolation='auto')
    
    ax.grid(linewidth=0.5, alpha=0.15)
    
    ax.set_facecolor('k')
    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Declination [deg]")
    ax.set_title(f"30 Dor C    {N_ob} Obs.    " + 
                 f"({Ebin[0]/1e3:.2f} - {Ebin[1]/1e3:.2f}) keV")
    
    cbar = fig.colorbar(raw_im, ax=ax)
    cbar.ax.set_ylabel("Counts")
    
    # Use sherpa to smooth the image (then display using imshow).
    sm_fig = plt.figure()
    sm_ax = sm_fig.add_subplot(projection=coordsys)
    
    smooth_arr = gsmooth(im_arr, sigma=1.5)
    sm_im = sm_ax.imshow(smooth_arr.T, origin='lower', cmap='hot',
                         norm=clrs.LogNorm(vmin=1))
    
    sm_ax.grid(linewidth=0.5, alpha=0.15)
    
    sm_ax.set_facecolor('k')
    sm_ax.set_xlabel("Right Ascension [deg]")
    sm_ax.set_ylabel("Declination [deg]")
    sm_ax.set_title(f"30 Dor C    {N_ob} Obs.    " + 
                 f"({Ebin[0]/1e3:.2f} - {Ebin[1]/1e3:.2f}) keV")
    
    sm_cbar = fig.colorbar(sm_im, ax=sm_ax)
    sm_cbar.ax.set_ylabel("Counts")
    plt.show()