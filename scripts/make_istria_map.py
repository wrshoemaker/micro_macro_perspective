
import warnings
import geopandas as gpd
import osmnx as ox
import contextily as ctx
import matplotlib.pyplot as plt
import folium
from shapely.geometry import mapping

# -----------------------------
# 1. Suppress RequestsDependencyWarning from requests (Python 3.12 safe)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message=".*Unable to find acceptable character detection dependency.*"
)

# -----------------------------
# 2. Get Istria polygon from OSM
# -----------------------------
istria = ox.geocode_to_gdf("Istria County, Croatia")

# -----------------------------
# 3. Project to a metric CRS for accurate centroid
# EPSG 32632 = UTM zone 32N, suitable for Istria
# -----------------------------
istria_proj = istria.to_crs(epsg=32632)

# -----------------------------
# 4. Compute centroid for map centering
# -----------------------------
centroid_proj = istria_proj.geometry.centroid.iloc[0]
# convert back to lat/lon for folium
centroid_latlon = gpd.GeoSeries([centroid_proj], crs=istria_proj.crs).to_crs(epsg=4326).iloc[0]

# -----------------------------
# 5. Static map (GeoPandas + Contextily)
# -----------------------------
istria_merc = istria.to_crs(epsg=3857)  # Web Mercator for basemap tiles

fig, ax = plt.subplots(figsize=(10, 10))
istria_merc.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)

# add terrain basemap (latest contextily provider)
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain, zoom=10)

ax.set_axis_off()
plt.tight_layout()
plt.savefig("istria_map.png", dpi=300, bbox_inches="tight")
plt.show()
