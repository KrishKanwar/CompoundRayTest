# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: ol-c-kernel
#     language: python
#     name: ol-c-kernel
# ---

# %% Project setup
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""


# %%
# plotting setup
# matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
from cmath import pi
# import pandas as pd

# %%
# transformation between Cartesian <-> spherical coordinates
from geometry import cart2sph, sph2cart

# read in csv data
data = np.genfromtxt('Projection2D/data_extraction_test_data.csv', delimiter=',')
pts = data[787:1552, 3:6]
ommatid_data = np.genfromtxt('Projection2D/ommatid_data.csv', delimiter=',')
plot_colors = ommatid_data[786:1552]/255.0
print(plot_colors)

# convert to spherical coordinate in [r=1, theta, phi] in radia n
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)

# convert to Cartesian
xyz = sph2cart(rtp)
# convert to spherical
rtp = cart2sph(xyz)

# %%
# Mollweide projections
from geometry import sph2Mollweide

# use rtp from earlier
# pick a few points
plot_pts = data[787:1552, 3:6]

print(plot_pts)

cart_plot_pts = cart2sph(plot_pts)

print(plot_pts)

# convert to spherical coordinate in [r=1, theta, phi] in radian
#rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
xy = sph2Mollweide(cart_plot_pts[:,1:3])

# plot
# define guidelines
ww = np.stack((np.linspace(0,180,19), np.repeat(-180,19)), axis=1)
w = np.stack((np.linspace(180,0,19), np.repeat(-90,19)), axis=1)
m = np.stack((np.linspace(0,180,19), np.repeat(0,19)), axis=1)
e = np.stack((np.linspace(180,0,19), np.repeat(90,19)), axis=1)
ee = np.stack((np.linspace(0,180,19), np.repeat(180,19)), axis=1)
pts = np.vstack((ww,w,m,e,ee))
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
meridians_xy = sph2Mollweide(rtp[:,1:3])

pts = np.stack((np.repeat(45,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
n45_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(90,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
eq_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(135,37), np.linspace(-180,180,37)), axis=1)
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
s45_xy = sph2Mollweide(rtp[:,1:3])

plt.plot(meridians_xy[:,0], meridians_xy[:,1], '-k', linewidth=1.0)
plt.plot(n45_xy[:,0], n45_xy[:,1], '-k', linewidth=1)
plt.plot(eq_xy[:,0], eq_xy[:,1], '-k', linewidth=1)
plt.plot(s45_xy[:,0], s45_xy[:,1], '-k', linewidth=1)
plt.scatter(xy[:,0], xy[:,1], c = plot_colors)
plt.xlabel("azimuth")
plt.ylabel("elevation")

plt.show()
