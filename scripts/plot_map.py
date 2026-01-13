import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from PIL import Image
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib_scalebar.scalebar import ScaleBar

import config



#area = gpd.read_file("%smap/Shape/NHDWaterbody.shp" % config.data_directory)
# EPSG:4269
#print(area.crs)


# get lat long points
#pond_metadata = pandas.read_csv("%s20130801_INPondDataMod.csv" % config.data_directory, sep=',')
#lats = pond_metadata.lat.values
#longs = pond_metadata.long.values


#from shapely.geometry import Point

#points = gpd.GeoDataFrame(
#    geometry=[Point(lon, lat) for lon, lat in zip(longs, lats)],
#    crs="EPSG:4326"   # WGS84 lat/lon
#)

#
#points = points.to_crs("EPSG:4269")

#fig, ax = plt.subplots(figsize=(8, 8))

#area.plot(ax=ax, color="lightgrey", edgecolor="black")
#points.plot(ax=ax, color="red", markersize=40)

#ax.set_title("Study area (EPSG:4269) with observation points")
#ax.set_axis_off()

#plt.show()

#fig.subplots_adjust(hspace=0.25, wspace=0.15)
#fig_name = "%smap.png" % config.analysis_directory
#fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
#plt.close()




sample_ponds = pd.read_csv("%s20130801_INPondDataMod.csv" % config.data_directory, sep=',')
#all_ponds = pd.read_csv("~/GitHub/DistDecay/map/RefugePonds.csv")

water = gpd.read_file("%smap/Shape/NHDWaterbody.shp" % config.data_directory)

# EPSG:4269
#print(water.crs) 

sample_ponds_gdf = gpd.GeoDataFrame(
    sample_ponds,
    geometry=gpd.points_from_xy(sample_ponds["long"], sample_ponds["lat"]),
    crs="EPSG:4269"
)

fig, ax = plt.subplots(figsize=(8, 8))

# Water bodies (geom_polygon)
water.plot(
    ax=ax,
    edgecolor=(0.12, 0.56, 1.0, 0.5),   # dodgerblue w/ alpha
    facecolor="skyblue",
    linewidth=0.1,
    alpha=0.3
)

# Sample ponds (blue)
sample_ponds_gdf.plot(
    ax=ax,
    color="blue",
    markersize=15
)

# Axis limits & labels
ax.set_xlim(-86.5, -86.1)
ax.set_ylim(38.94, 39.22)
ax.set_xlabel("Longitude", fontsize=18)
ax.set_ylabel("Latitude", fontsize=18)

# Style (ggplot theme equivalent)
ax.tick_params(labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(3)

ax.set_aspect(1.3)

scalebar = ScaleBar(
    dx=1,
    units="km",
    location="lower right",
    scale_loc="bottom"
)
ax.add_artist(scalebar)

# North arrow
ax.annotate(
    "N",
    xy=(0.05, 0.1),
    xytext=(0.05, 0.2),
    arrowprops=dict(facecolor="black", width=5, headwidth=15),
    ha="center",
    va="center",
    fontsize=14,
    xycoords=ax.transAxes
)

inset_ax = fig.add_axes([0.15, 0.68, 0.25, 0.25])

water.plot(ax=inset_ax, color="gray", edgecolor="black")
#all_ponds_gdf.plot(ax=inset_ax, color="red", markersize=5)
sample_ponds_gdf.plot(ax=inset_ax, color="blue", markersize=5)

inset_ax.set_axis_off()


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%smap.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()