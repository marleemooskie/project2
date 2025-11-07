#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a map visualization of the sites I am currently using.
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import dataframe of sites
sites_df = pd.read_csv('/Users/marleeyork/Documents/project2/data/site_locations_for_PRISM.csv')
sites_df.Lon = sites_df.Lon.astype('float')
sites_df.Lat = sites_df.Lat.astype('float')

# Deciding which of the sites I am going to plot
# df is coming from QAQC.py
site_mask = sites_df['Site'].isin(df['Site'].unique())
sites_df = sites_df[site_mask]

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection=ccrs.LambertConformal())

# Show both US and Canada
ax.set_extent([-140, -50, 20, 75], ccrs.Geodetic())

# Base map features
ax.add_feature(cfeature.LAND, facecolor='white')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor='gray')
ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray')
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='black')

# Plot site points
ax.scatter(
    sites_df['Lon'],
    sites_df['Lat'],
    color='red',
    s=40,
    transform=ccrs.PlateCarree(),
    zorder=5,
    label='Site locations'
)

# Optional: label sites
for i, row in sites_df.iterrows():
    ax.text(
        row['Lon'] + 1 * (1 if i % 2 == 0 else -1),
        row['Lat'] + 1 * (1 if i % 3 == 0 else -1),
        row['Site'],
        transform=ccrs.PlateCarree(),
        fontsize=8,
        ha='center',
        va='center'
    )

plt.title("Locations of the sites I'm starting with")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
