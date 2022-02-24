#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:19:22 2022

Compute Steri height - example

@author: ccamargo
"""

# Import libraries
import xarray as xr
import numpy as np
import gsw  # TEOS-10 software
import matplotlib.pyplot as plt
from cmocean import cm as cm

#%% Example from the gsw documentation
# http://www.teos-10.org/pubs/gsw/html/gsw_geo_strf_steric_height.html
dyn_h = gsw.geo_strf_dyn_height([34.7118, 34.7324], [28.8099, 4.3236], [10, 1000], 1000)
dyn_h / 9.7963
SA = [
    34.7118,
    34.8915,
    35.0256,
    34.8472,
    34.7366,
    34.7324,
]
CT = [
    28.8099,
    28.4392,
    22.7862,
    10.2262,
    6.8272,
    4.3236,
]
p = [10, 50, 125, 250, 600, 1000]
p_ref = 1000

dyn_h = gsw.geo_strf_dyn_height(SA, CT, p, p_ref)
steric = dyn_h / 9.7963

#%% get EN4 example data
# data from https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html
path = "/Volumes/LaCie_NIOZ/data/steric/ocean_TS/EN4/EN.4.2.2.analyses.g10.2021/"
file = "EN.4.2.2.f.analysis.g10.202106.nc"
ds = xr.open_dataset(path + file)
print(ds)
# ds.temperature.mean(dim=('lat','lon')).plot()
# ds.temperature[:,0,:,:].mean(dim=('lat','lon')).plot()


#%%
lat = np.array(ds.lat)
lon = np.array(ds.lon)
llon, llat = np.meshgrid(lon, lat)

# make land mask
mask = np.array(ds.salinity[-1, 2, :, :])
mask[np.isfinite(mask)] = 1
mask[np.where(llat > 66)] = np.nan
mask[np.where(llat < -66)] = np.nan
plt.pcolor(mask)
plt.title("land mask")
plt.show()
# if we want only to max 2000
# ds = ds.where(ds.depth<2001, drop=True)
depth = np.array(ds.depth)

# get dimensions
dimtime, dimdepth, dimlat, dimlon = np.array(ds.temperature).shape

# %%check how max depth looks like:
salinity = np.array(ds.salinity[0] * mask).reshape(dimdepth, dimlat * dimlon)
max_depth = np.array(
    [
        depth[np.where(np.isnan(salinity[:, icoord]))[0][0]]
        if np.any(np.isnan(salinity[:, icoord]))
        else depth[-1]
        for icoord in range(dimlat * dimlon)
    ]
).reshape(dimlat, dimlon)
# plot
plt.figure()
plt.pcolor(
    max_depth, cmap=cm.deep, vmin=np.nanmin(max_depth), vmax=np.nanmax(max_depth)
)
plt.colorbar(label="depth (m)")
plt.title("Maximum local depth")
plt.show()


#%% compute steric height at each grid point at each timestep


# loop over time (in case you have several timesteps)
# for itime in range(len(ds.time)):
itime = 0

# get  Salinity and Temperature
salinity = np.array(ds.salinity[itime] * mask).reshape(dimdepth, dimlat * dimlon)
# Note: Temperature is in Kelvins! Transform in Celsius:
temperature = np.array((ds.temperature[itime] - 273.15) * mask).reshape(
    dimdepth, dimlat * dimlon
)

# make an empty array for steric height
ste_h = np.full_like(np.zeros((dimtime, dimdepth, dimlat * dimlon)), np.nan)

llat = llat.flatten()
llon = llon.flatten()
mask = mask.flatten()
#% %
for icoord in range(dimlat * dimlon):
    if mask[icoord] == 1:  # if it's ocean

        # compute pressure from depth
        p = gsw.p_from_z(-depth, llat[icoord])

        # compure absolute salinity from pratical salinity
        SP = salinity[:, icoord]
        SA = gsw.SA_from_SP(SP, p, llon[icoord], llat[icoord])

        # compute consevative temperature form in situ temperature
        T = temperature[:, icoord]
        CT = gsw.CT_from_t(SA, T, p)

        # find valid index
        idx = np.isfinite(CT)

        # reference pressure
        p_ref = p[idx].max()  # in relation to maximum local depth
        # Note: if you want only in relation to max 2000m, you need to state it!

        # compute dynamic height
        dyn_h = gsw.geo_strf_dyn_height(SA[idx], CT[idx], p[idx], p_ref)

        # compute steric height
        i = np.argwhere(idx)[0, 0]
        # j=np.argwhere(idx)[-1,0]+1
        j = min(len(dyn_h), np.argwhere(idx)[-1, 0] + 1)
        ste_h[itime, i:j, icoord] = dyn_h / 9.7963


#%% visualize it:
plt.figure()
plt.title("Steric SL")
plt.pcolor(
    ste_h[0, 0, :].reshape(dimlat, dimlon),
    cmap=cm.thermal,
    vmin=-2,
    vmax=5,
)
plt.colorbar(label="steric height (m)")
plt.show()
