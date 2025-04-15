import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
import math

csv_file = "Final_Fire_Incidence_Data_with_PopDensity.csv"
df = pd.read_csv(csv_file)

df["geometry"] = df["the_geom"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf = gdf.dropna(subset=["geometry"])
gdf = gdf[gdf.geometry.is_valid]

gdf["high_risk_count"] = gdf["high_risk_count"].fillna(0)
gdf["medium_risk_count"] = gdf["medium_risk_count"].fillna(0)
gdf["low_risk_count"] = gdf["low_risk_count"].fillna(0)
gdf["area_mile2"] = gdf["area_mile2"].fillna(0)
gdf["pop_density_per_sqmi"] = gdf["pop_density_per_sqmi"].fillna(0)

gdf["weight"] = ((
    3*gdf["high_risk_count"]
  + 2*gdf["medium_risk_count"]
  + 1*gdf["low_risk_count"])
  *   gdf["area_mile2"]
  *   gdf["pop_density_per_sqmi"]
)

gdf_3857 = gdf.to_crs(epsg=3857)
gdf_3857["centroid"] = gdf_3857.geometry.centroid

cx = gdf_3857["centroid"].x.values
cy = gdf_3857["centroid"].y.values
weights = gdf_3857["weight"].values

buffer_m = 100
xmin, ymin, xmax, ymax = gdf_3857.total_bounds
xmin -= buffer_m
ymin -= buffer_m
xmax += buffer_m
ymax += buffer_m

num_x = 300
num_y = 300
xs = np.linspace(xmin, xmax, num_x)
ys = np.linspace(ymin, ymax, num_y)
X, Y = np.meshgrid(xs, ys)
def epanechnikov_2d(distance, h):
    mask = distance < h
    c = 2.0 / (math.pi * h**2)
    out = np.zeros_like(distance, dtype=float)
    out[mask] = c * (1.0 - (distance[mask]**2 / h**2))
    return out

bandwidth = 2500.0

X3 = X[..., None]
Y3 = Y[..., None]
cx_ = cx[None, :]
cy_ = cy[None, :]

dist = np.sqrt((X3 - cx_)**2 + (Y3 - cy_)**2)
Kvals = epanechnikov_2d(dist, bandwidth)

weighted = Kvals * weights
density = weighted.sum(axis=2)


fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.epsg(3857))

ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray")
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5)
ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.5)

zip_polygons = ShapelyFeature(
    gdf_3857["geometry"],
    ccrs.epsg(3857),
    edgecolor="black",
    facecolor="none",
    linewidth=0.5
)
ax.add_feature(zip_polygons)

im = ax.imshow(
    density,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    alpha=0.5,
    cmap="jet",
    transform=ccrs.epsg(3857)
)
cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)
cbar.set_label("Epanechnikov KDE (Weighted)")

ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.epsg(3857))
plt.title("NYC Zip Codes + Weighted Epanechnikov KDE")
plt.show()

points_list = []
dens_list = []

for i in range(num_y):
    for j in range(num_x):
        xcoord = xs[j]
        ycoord = ys[i]
        points_list.append(Point(xcoord, ycoord))
        dens_list.append(density[i, j])

gdf_points = gpd.GeoDataFrame(
    {"density": dens_list},
    geometry=points_list,
    crs="EPSG:3857"
)

joined = gpd.sjoin(
    gdf_points,
    gdf_3857,
    how="left",
    predicate="within"
)

grouped = joined.groupby("MODZCTA")["density"]

mean_density_by_zip = grouped.mean()

val_min = mean_density_by_zip.min()
val_max = mean_density_by_zip.max()
range_ = val_max - val_min

scaled = 100 * (mean_density_by_zip - val_min) / range_

df_zip_density = pd.DataFrame({
    "MODZCTA": mean_density_by_zip.index,
    "mean_kde": mean_density_by_zip.values,
    "scaled_kde_0_100": scaled.values
})

df_zip_density.sort_values("mean_kde", ascending=False, inplace=True)

output_csv = "kde_by_zipcode.csv"
df_zip_density.to_csv(output_csv, index=False)

print(f"Exported KDE by ZIP code to '{output_csv}'. Top rows:")
print(df_zip_density.head(10))
