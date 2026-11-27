import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import os


ob_dirs = [direc for direc in os.scandir("../Data")]

N_ob = len(os.listdir("../Data"))

s_x = np.array([])
s_y = np.array([])

s_ra = np.array([])
s_dec = np.array([])

s_energies = np.array([])
s_grades = np.array([])

for cur_ob in range(N_ob):
    cur_ob_files = [file for file in os.scandir(ob_dirs[cur_ob])]
    
    # Current event and image file.
    cur_ev_file = cur_ob_files[0]
    cur_im_file = cur_ob_files[1]
    
    # Check that the event and image files correspond to the same observation.
    if cur_ev_file.name[5:10] != cur_im_file.name[5:10]:
        print("WARNING: File Mismatch!")
    
    cur_ev_fits = fits.open(cur_ev_file)
    cur_im_fits = fits.open(cur_im_file)
    
    # Take useful data from the fits file.
    obs_id = cur_ev_fits["PRIMARY"].header["OBS_ID"]
    x_vals = cur_ev_fits[1].data["x"]
    y_vals = cur_ev_fits[1].data["y"]
    energies = cur_ev_fits[1].data["energy"]
    
    offset_x = cur_im_fits[0].header["CRVAL1P"]
    offset_y = cur_im_fits[0].header["CRVAL2P"]
    
    scale_x = cur_im_fits[0].header["CDELT1P"]
    scale_y = cur_im_fits[0].header["CDELT2P"]
    
    # Correct for a pixel-offset in the fits file.
    
    x_vals = (x_vals - offset_x)/scale_x
    y_vals = (y_vals - offset_y)/scale_y
    
    
    # Split the image into multiple sequential energy bands.
    E_range = [2e3, 8e3]
    splits = 2
    energy_bin_edges = np.linspace(E_range[0], E_range[1], splits)
    energy_bins = np.round([[energy_bin_edges[i], energy_bin_edges[i+1]] 
                            for i in range(splits-1)], 3)
    
    # Get the coordinate system of the fits file.
    coordsys = WCS(header=cur_im_fits[0].header)
    #Convert pixel values to sky-coordinates.
    skycoords = coordsys.pixel_to_world(x_vals, y_vals)
    
    # Save the sky-coordinates as separate RA and DEC arrays.
    ra_vals = np.array([coord[0] for coord in skycoords.to_table().as_array()])
    dec_vals = np.array([coord[1] for coord in skycoords.to_table().as_array()])
    
    # Extract the grade of each event.
    grades = cur_ev_fits[1].data["grade"]
    
    s_x = np.append(s_x, x_vals)
    s_y = np.append(s_y, y_vals)
    s_ra = np.append(s_ra, ra_vals)
    s_dec = np.append(s_dec, dec_vals)
    s_energies = np.append(s_energies, energies)
    s_grades = np.append(s_grades, grades)

for Ebin in energy_bins:
    # Only accept events with a grade below or equal to 4.
    graded_x = s_x[s_grades <= 4]
    graded_y = s_y[s_grades <= 4]
    graded_ra = s_ra[s_grades <= 4]
    graded_dec = s_dec[s_grades <= 4]
    graded_energies = s_energies[s_grades <= 4]
    
    # Partition events into energy bins.
    x_vals_Ecut = graded_x[(graded_energies >= Ebin[0]) & (graded_energies<= Ebin[1])]
    y_vals_Ecut = graded_y[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    ra_vals_Ecut = graded_ra[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    dec_vals_Ecut = graded_dec[(graded_energies >= Ebin[0]) & (graded_energies <= Ebin[1])]
    
    # Plot the 2D histogram, an image of 30 Dor C.
    fig, ax = plt.subplots()
    im = ax.hist2d(ra_vals_Ecut, dec_vals_Ecut, bins=400, cmap="hot", norm=clrs.LogNorm())
    
    # Correct for the fact that these coordinates are projected on a sphere.
    # Note that this aspect correction is an mean approximation which should work for small angles.
    ax.set_aspect(1/np.sin(np.pi/2 - np.mean([np.amax(dec_vals_Ecut), np.amin(dec_vals_Ecut)])
                           *np.pi/180))
    ax.xaxis.set_inverted(True)
    
    ax.grid(linewidth=0.5, alpha=0.5)
    
    ax.set_facecolor('k')
    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Declination [deg]")
    ax.set_title(f"30 Dor C    {N_ob} Obs.    " + 
                 f"({Ebin[0]/1e3:.2f} - {Ebin[1]/1e3:.2f}) keV")
    
    cbar = fig.colorbar(im[3], ax=ax)
    cbar.ax.set_ylabel("Counts")

    plt.show()