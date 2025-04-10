import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import unary_union

zip_csv = "Final_Fire_Incidence_Data_with_PopDensity copy.csv"
zip_df = pd.read_csv(zip_csv)

zip_df["geometry"] = zip_df["the_geom"].apply(wkt.loads)

gdf_zip = gpd.GeoDataFrame(zip_df, geometry="geometry", crs="EPSG:4326")
gdf_zip = gdf_zip.dropna(subset=["geometry"])
gdf_zip = gdf_zip[gdf_zip.geometry.is_valid]

gdf_zip_3857 = gdf_zip.to_crs(epsg=3857)

gdf_zip_3857["zip_area_m2"] = gdf_zip_3857.geometry.area

fire_csv = "FDNY_Firehouse_Listing_20250316 copy.csv"
fire_df = pd.read_csv(fire_csv)

gdf_fire = gpd.GeoDataFrame(
    fire_df,
    geometry=gpd.points_from_xy(fire_df.Longitude, fire_df.Latitude),
    crs="EPSG:4326"
)

gdf_fire_3857 = gdf_fire.to_crs(epsg=3857)

gdf_fire_3857["buffer_0_5km"] = gdf_fire_3857.geometry.buffer(500)
gdf_fire_3857["buffer_1km"] = gdf_fire_3857.geometry.buffer(1000)
gdf_fire_3857["buffer_1_5km"] = gdf_fire_3857.geometry.buffer(1500)

gdf_fire_3857["doughnut_0_5_1"] = gdf_fire_3857["buffer_1km"].difference(gdf_fire_3857["buffer_0_5km"])
gdf_fire_3857["doughnut_1_1_5"] = gdf_fire_3857["buffer_1_5km"].difference(gdf_fire_3857["buffer_1km"])

circles_0_5 = gdf_fire_3857.copy().set_geometry("buffer_0_5km")
doughnuts_0_5_1 = gdf_fire_3857.copy().set_geometry("doughnut_0_5_1")
doughnuts_1_1_5 = gdf_fire_3857.copy().set_geometry("doughnut_1_1_5")

results = []

for idx, zip_row in gdf_zip_3857.iterrows():
    modzcta = zip_row["MODZCTA"] if "MODZCTA" in zip_row else idx
    zip_poly = zip_row["geometry"]
    zip_area = zip_row["zip_area_m2"]

    circles_in_zip_0_5 = circles_0_5[circles_0_5.geometry.intersects(zip_poly)]["buffer_0_5km"]
    if circles_in_zip_0_5.empty:
        coverage_area_0_5 = 0.0
    else:
        union_0_5 = circles_in_zip_0_5.unary_union
        intersect_0_5 = union_0_5.intersection(zip_poly)
        coverage_area_0_5 = intersect_0_5.area
    coverage_pct_0_5 = 100.0 * coverage_area_0_5 / zip_area if zip_area > 0 else 0.0

    circles_list = list(circles_in_zip_0_5)
    overlap_polys = []
    if len(circles_list) < 2:
        overlap_area = 0.0
    else:
        for i in range(len(circles_list) - 1):
            for j in range(i + 1, len(circles_list)):
                inter = circles_list[i].intersection(circles_list[j])
                if not inter.is_empty:
                    overlap_polys.append(inter)
        if overlap_polys:
            overlap_union = unary_union(overlap_polys)
            overlap_clipped = overlap_union.intersection(zip_poly)
            overlap_area = overlap_clipped.area
        else:
            overlap_area = 0.0
    overlap_pct = 100.0 * overlap_area / zip_area if zip_area > 0 else 0.0

    doughnuts_in_zip_0_5_1 = doughnuts_0_5_1[doughnuts_0_5_1.geometry.intersects(zip_poly)]["doughnut_0_5_1"]
    if doughnuts_in_zip_0_5_1.empty:
        coverage_area_donut_0_5_1 = 0.0
    else:
        union_donut_0_5_1 = doughnuts_in_zip_0_5_1.unary_union
        intersect_donut_0_5_1 = union_donut_0_5_1.intersection(zip_poly)
        coverage_area_donut_0_5_1 = intersect_donut_0_5_1.area
    coverage_pct_donut_0_5_1 = 100.0 * coverage_area_donut_0_5_1 / zip_area if zip_area > 0 else 0.0

    doughnuts_in_zip_1_1_5 = doughnuts_1_1_5[doughnuts_1_1_5.geometry.intersects(zip_poly)]["doughnut_1_1_5"]
    if doughnuts_in_zip_1_1_5.empty:
        coverage_area_donut_1_1_5 = 0.0
    else:
        union_donut_1_1_5 = doughnuts_in_zip_1_1_5.unary_union
        intersect_donut_1_1_5 = union_donut_1_1_5.intersection(zip_poly)
        coverage_area_donut_1_1_5 = intersect_donut_1_1_5.area
    coverage_pct_donut_1_1_5 = 100.0 * coverage_area_donut_1_1_5 / zip_area if zip_area > 0 else 0.0

    results.append({
        "MODZCTA": modzcta,
        "zip_area_m2": zip_area,
        "high_risk_coverage_area": coverage_area_0_5,
        "high_risk_coverage_area_pct": coverage_pct_0_5,
        "high_risk_intersection_area": overlap_area,
        "high_risk_intersection_area_pct": overlap_pct,
        "medium_risk_coverage_area": coverage_area_donut_0_5_1,
        "medium_risk_coverage_area_pct": coverage_pct_donut_0_5_1,
        "low_risk_coverage_area": coverage_area_donut_1_1_5,
        "low_risk_coverage_area_pct": coverage_pct_donut_1_1_5
    })

coverage_df = pd.DataFrame(results)
coverage_df = coverage_df.sort_values("MODZCTA", ascending=True)

print("=== Buffer Coverage by ZIP Code ===")
print(coverage_df)

output_csv = "all_zip_full_buffer_coverage_with_overlap.csv"
coverage_df.to_csv(output_csv, index=False)
print(f"Results exported to '{output_csv}'.")
