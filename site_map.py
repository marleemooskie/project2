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
sites_df = pd.read_csv('/Users/marleeyork/Documents/project2/data/starter_sites_list.csv')
sites_df.lon = sites_df.lon.astype('float')
sites_df.lat = sites_df.lat.astype('float')

# Set up map
fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-125, -66.5, 24, 49], ccrs.Geodetic())

# Base map
ax.add_feature(cfeature.LAND, facecolor='white')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# Plot sites
ax.scatter(
    sites_df['lon'],
    sites_df['lat'],
    color='red',
    s=40,
    transform=ccrs.PlateCarree(),
    zorder=5,
    label='Site locations'
)

# Optionally label sites
for i, row in sites_df.iterrows():
    ax.text(
        row['lon'] + 1.2*(1 if i % 2 == 0 else -1),  # alternate left/right
        row['lat'] + 1.2*(1 if i % 3 == 0 else -1),  # stagger vertically
        row['site'],
        transform=ccrs.PlateCarree(),
        fontsize=8,
        ha='center',
        va='center'
    )

plt.title("Starter Sites I'm Using")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

