# %%
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import unary_union
from deap import base, creator, tools
import itertools
import time


try:
    from shapely.ops import union_all
except ImportError:
    def union_all(geoms):
        return unary_union(geoms)

# %%
# -------------------- Function Definitions --------------------

def compute_FRpi(row):
    risk_sum = 3 * row.get("num_high_risk", 0) + 2 * row.get("num_med_risk", 0) + 1 * row.get("num_low_risk", 0)
    pop_dens = row.get("pop_density_per_sqkm", 0)
    zip_area = row["zip_area_m2"]
    return risk_sum * (risk_sum / zip_area) * pop_dens if zip_area > 0 else 0.0

def mateNoDuplicates(ind1, ind2):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2
    tools.cxTwoPoint(ind1, ind2)
    fixDuplicates(ind1, pot_indices)
    fixDuplicates(ind2, pot_indices)
    return ind1, ind2

def fixDuplicates(individual, valid_indices):
    used = set()
    duplicates = []
    for i, val in enumerate(individual):
        if val in used:
            duplicates.append(i)
        else:
            used.add(val)
    if not duplicates:
        return
    unused = list(set(valid_indices) - used)
    random.shuffle(unused)
    for d in duplicates:
        if not unused:
            break
        individual[d] = unused.pop()

def mutReplaceNoDuplicates(ind):
    if len(ind) > 0:
        pos = random.randrange(len(ind))
        ind[pos] = random.choice(pot_indices)
        fixDuplicates(ind,pot_indices)
    return (ind,)

def evaluate(individual, W_OL, W_OM, W_OH, baseline_dict):
    new_stations = gdf_pot.iloc[individual].copy()
    new_stations["buffer_0_5km"] = new_stations.geometry.buffer(RADIUS_INNER)
    new_stations["buffer_1km"] = new_stations.geometry.buffer(RADIUS_MEDIUM)
    new_stations["buffer_1_5km"] = new_stations.geometry.buffer(RADIUS_OUTER)
    new_stations["doughnut_0_5_1"] = new_stations["buffer_1km"].difference(new_stations["buffer_0_5km"])
    new_stations["doughnut_1_1_5"] = new_stations["buffer_1_5km"].difference(new_stations["buffer_1km"])

    total_score = 0.0
    for idx, zip_row in gdf_risk.iterrows():
        zip_poly = zip_row.geometry
        FRpi = zip_row["FRpi"]
        modzcta = zip_row["MODZCTA"]

        high_cov = union_all(new_stations[new_stations["buffer_0_5km"].intersects(zip_poly)]["buffer_0_5km"]).intersection(zip_poly).area if not new_stations[new_stations["buffer_0_5km"].intersects(zip_poly)].empty else 0.0
        med_cov = union_all(new_stations[new_stations["doughnut_0_5_1"].intersects(zip_poly)]["doughnut_0_5_1"]).intersection(zip_poly).area if not new_stations[new_stations["doughnut_0_5_1"].intersects(zip_poly)].empty else 0.0
        low_cov = union_all(new_stations[new_stations["doughnut_1_1_5"].intersects(zip_poly)]["doughnut_1_1_5"]).intersection(zip_poly).area if not new_stations[new_stations["doughnut_1_1_5"].intersects(zip_poly)].empty else 0.0

        candidate_cov = W_OH * high_cov + W_OM * med_cov + W_OL * low_cov
        baseline_score = baseline_dict.get(modzcta, 0.0)
        total_score += baseline_score + candidate_cov + FRpi

    return (total_score,)
# %%
# -------------------- Parameters and Setup --------------------

W_OL_values = [3]#, 2 , 3, 4, 5]
W_OM_values = [2]#, 2 , 3, 4, 5]
W_OH_values = [1]#1, 2 , 3, 4, 5]

NNEW = 12
RADIUS_INNER = 500
RADIUS_MEDIUM = 1000
RADIUS_OUTER = 1500
VERBOSE = False

baseline_csv = "Data/all_zip_full_buffer_coverage_with_overlap.csv"
df_baseline = pd.read_csv(baseline_csv)

risk_csv = "Data/Merged_Risk_PopDensity_SquareKilometers.csv"
risk_df = pd.read_csv(risk_csv)
risk_df["geometry"] = risk_df["the_geom"].apply(wkt.loads)
gdf_risk = gpd.GeoDataFrame(risk_df, geometry="geometry", crs="EPSG:4326")
gdf_risk = gdf_risk.dropna(subset=["geometry"])
gdf_risk = gdf_risk[gdf_risk.geometry.is_valid]
gdf_risk = gdf_risk.to_crs(epsg=3857)
gdf_risk["zip_area_m2"] = gdf_risk.geometry.area
gdf_risk["FRpi"] = gdf_risk.apply(compute_FRpi, axis=1)

pot_df = pd.read_csv("Data/Potential_location.csv")
gdf_pot = gpd.GeoDataFrame(pot_df, geometry=gpd.points_from_xy(pot_df.longitude, pot_df.latitude), crs="EPSG:4326")
gdf_pot = gdf_pot.to_crs(epsg=3857)
pot_indices = list(range(len(gdf_pot)))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual(random.sample(pot_indices, NNEW)))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", mateNoDuplicates)
toolbox.register("mutate", mutReplaceNoDuplicates)
toolbox.register("select", tools.selTournament, tournsize=3)

# %%
# -------------------- Main Function --------------------

def main():
    results = []
    for W_OL, W_OM, W_OH in itertools.product(W_OL_values, W_OM_values, W_OH_values):
        start_time = time.time()
        print(f"Iteration: W_OL: {W_OL}, W_OM: {W_OM}, W_OH: {W_OH}")

        random.seed(42)

        # Create a fresh toolbox for this run
        toolbox = base.Toolbox()
        toolbox.register("individual", lambda: creator.Individual(random.sample(pot_indices, NNEW)))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", mateNoDuplicates)
        toolbox.register("mutate", mutReplaceNoDuplicates)
        toolbox.register("select", tools.selTournament, tournsize=3)

        df_baseline["baseline_score"] = (
            W_OH * df_baseline["high_risk_coverage_area"] +
            W_OM * df_baseline["medium_risk_coverage_area"] +
            W_OL * df_baseline["low_risk_coverage_area"]
        )
        baseline_dict = dict(zip(df_baseline["MODZCTA"], df_baseline["baseline_score"]))

        def fitness(ind):
            return evaluate(ind, W_OL, W_OM, W_OH, baseline_dict)

        toolbox.register("evaluate", fitness)
        
        population = toolbox.population(n=20)
        NGEN = 10
        CXPB = 0.7
        MUTPB = 0.8

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        global_best = tools.selBest(population, 1)[0]

        for gen in range(NGEN):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population[:] = offspring

            current_best = tools.selBest(population, 1)[0]
            if current_best.fitness.values[0] > global_best.fitness.values[0]:
                global_best = toolbox.clone(current_best)

        best_indices = list(global_best)
        result_df = pot_df.iloc[best_indices].copy()
        result_df["incidi"] = best_indices
        result_df = result_df[["incidi", "latitude", "longitude"]]
        result_df["W_OL"] = W_OL
        result_df["W_OM"] = W_OM
        result_df["W_OH"] = W_OH
        result_df["zipcode"] = pot_df.iloc[best_indices]["zipcode"].values

        results.append(result_df)

        elapsed_time = time.time() - start_time
        print(result_df)
        print(f"Completed in {elapsed_time:.2f} seconds\n")

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv("all_weight_combos_results_2.csv", index=False)
    return final_df


if __name__ == "__main__":
    final_results = main()

# %%
