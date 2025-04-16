import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import unary_union
import multiprocessing
import os
from math import comb
from itertools import combinations

try:
    from shapely.ops import union_all
except ImportError:
    def union_all(geoms):
        return unary_union(geoms)
from deap import base, creator, tools

W_OL = 75
W_OM = 75
W_OH = 100
NNEW = 8

RADIUS_INNER = 500
RADIUS_MEDIUM = 1000
RADIUS_OUTER = 1500

VERBOSE = True

baseline_csv = "all_zip_full_buffer_coverage_with_overlap.csv"
df_baseline = pd.read_csv(baseline_csv)
df_baseline["baseline_score"] = (
        W_OH * df_baseline["high_risk_coverage_area"] +
        W_OM * df_baseline["medium_risk_coverage_area"] +
        W_OL * df_baseline["low_risk_coverage_area"]
)
baseline_dict = dict(zip(df_baseline["MODZCTA"], df_baseline["baseline_score"]))
baseline_high_lookup = dict(zip(df_baseline["MODZCTA"], W_OH * df_baseline["high_risk_coverage_area"]))

total_baseline_fitness = sum(baseline_dict.values())

risk_csv = "Merged_Risk_PopDensity_SquareKilometers.csv"
risk_df = pd.read_csv(risk_csv)
risk_df["geometry"] = risk_df["the_geom"].apply(wkt.loads)
gdf_risk = gpd.GeoDataFrame(risk_df, geometry="geometry", crs="EPSG:4326")
gdf_risk = gdf_risk.dropna(subset=["geometry"])
gdf_risk = gdf_risk[gdf_risk.geometry.is_valid]
gdf_risk = gdf_risk.to_crs(epsg=3857)
gdf_risk["zip_area_m2"] = gdf_risk.geometry.area

def compute_FRpi(row):
    num_hr = row.get("num_high_risk", 0)
    num_mr = row.get("num_med_risk", 0)
    num_lr = row.get("num_low_risk", 0)
    risk_sum = W_OH * num_hr + W_OM * num_mr + W_OL * num_lr
    pop_dens = row.get("pop_density_per_sqkm", 0)
    zip_area = row["zip_area_m2"]
    if zip_area > 0:
        return risk_sum * (risk_sum / zip_area) * pop_dens
    return 0.0

gdf_risk["FRpi"] = gdf_risk.apply(compute_FRpi, axis=1)

pot_df = pd.read_csv("Potential_location.csv")
gdf_pot = gpd.GeoDataFrame(
    pot_df,
    geometry=gpd.points_from_xy(pot_df.longitude, pot_df.latitude),
    crs="EPSG:4326"
)
gdf_pot = gdf_pot.to_crs(epsg=3857)
pot_indices = list(range(len(gdf_pot)))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
def initUniqueIndices(icls, indices, n):
    return icls(random.sample(indices, n))
toolbox.register("individual", initUniqueIndices, creator.Individual, pot_indices, NNEW)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

def mutReplaceNoDuplicates(individual):
    if len(individual) > 0:
        pos = random.randrange(len(individual))
        individual[pos] = random.choice(pot_indices)
        fixDuplicates(individual, pot_indices)
    return (individual,)

toolbox.register("mate", mateNoDuplicates)
toolbox.register("mutate", mutReplaceNoDuplicates)

def evaluate(individual):
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
        buffers_high = new_stations[new_stations["buffer_0_5km"].intersects(zip_poly)]["buffer_0_5km"]
        if not buffers_high.empty:
            union_high = union_all(buffers_high.tolist())
            high_cov = union_high.intersection(zip_poly).area
        else:
            high_cov = 0.0
        buffers_med = new_stations[new_stations["doughnut_0_5_1"].intersects(zip_poly)]["doughnut_0_5_1"]
        if not buffers_med.empty:
            union_med = union_all(buffers_med.tolist())
            med_cov = union_med.intersection(zip_poly).area
        else:
            med_cov = 0.0
        buffers_low = new_stations[new_stations["doughnut_1_1_5"].intersects(zip_poly)]["doughnut_1_1_5"]
        if not buffers_low.empty:
            union_low = union_all(buffers_low.tolist())
            low_cov = union_low.intersection(zip_poly).area
        else:
            low_cov = 0.0
        candidate_cov = W_OH * high_cov + W_OM * med_cov + W_OL * low_cov
        baseline_score = baseline_dict.get(modzcta, 0.0)
        zip_score = baseline_score + candidate_cov + FRpi
        total_score += zip_score
    if VERBOSE:
        print(f"Evaluated candidate {individual} => total_score {total_score:.2f}")
    return (total_score,)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_with_breakdown(individual):

    new_stations = gdf_pot.iloc[individual].copy()
    new_stations["buffer_0_5km"] = new_stations.geometry.buffer(RADIUS_INNER)
    new_stations["buffer_1km"] = new_stations.geometry.buffer(RADIUS_MEDIUM)
    new_stations["buffer_1_5km"] = new_stations.geometry.buffer(RADIUS_OUTER)
    new_stations["doughnut_0_5_1"] = new_stations["buffer_1km"].difference(new_stations["buffer_0_5km"])
    new_stations["doughnut_1_1_5"] = new_stations["buffer_1_5km"].difference(new_stations["buffer_1km"])
    total_score = 0.0
    total_high_candidate = 0.0
    total_baseline_high = 0.0
    for idx, zip_row in gdf_risk.iterrows():
        zip_poly = zip_row.geometry
        FRpi = zip_row["FRpi"]
        modzcta = zip_row["MODZCTA"]
        baseline_score = baseline_dict.get(modzcta, 0.0)
        # Evaluate high risk coverage candidate:
        buffers_high = new_stations[new_stations["buffer_0_5km"].intersects(zip_poly)]["buffer_0_5km"]
        if not buffers_high.empty:
            union_high = union_all(buffers_high.tolist())
            high_cov = union_high.intersection(zip_poly).area
        else:
            high_cov = 0.0
        candidate_high = W_OH * high_cov
        total_high_candidate += candidate_high
        # Get baseline high risk coverage for this modzcta.
        baseline_high = baseline_high_lookup.get(modzcta, 0.0)
        total_baseline_high += baseline_high
        # Evaluate medium and low risk (for overall fitness, not needed in breakdown)
        buffers_med = new_stations[new_stations["doughnut_0_5_1"].intersects(zip_poly)]["doughnut_0_5_1"]
        if not buffers_med.empty:
            union_med = union_all(buffers_med.tolist())
            med_cov = union_med.intersection(zip_poly).area
        else:
            med_cov = 0.0
        buffers_low = new_stations[new_stations["doughnut_1_1_5"].intersects(zip_poly)]["doughnut_1_1_5"]
        if not buffers_low.empty:
            union_low = union_all(buffers_low.tolist())
            low_cov = union_low.intersection(zip_poly).area
        else:
            low_cov = 0.0
        candidate_cov = candidate_high + W_OM * med_cov + W_OL * low_cov
        zip_score = baseline_score + candidate_cov + FRpi
        total_score += zip_score
    return total_score, total_high_candidate, total_baseline_high

def brute_force_verification():

    total_combinations = comb(len(pot_indices), NNEW)
    threshold = 10000
    if total_combinations > threshold:
        print(f"Skipping brute force verification: {total_combinations} combinations exceeds threshold {threshold}.")
        return None, None
    if VERBOSE:
        print(f"Total combinations to evaluate: {total_combinations}")
    best_score = -np.inf
    best_candidate = None
    progress_interval = max(total_combinations // 10, 1)
    count = 0
    for candidate in combinations(pot_indices, NNEW):
        count += 1
        if VERBOSE and count % progress_interval == 0:
            print(f"Brute force progress: {count}/{total_combinations} candidates processed.")
        candidate = list(candidate)
        score = evaluate(candidate)[0]
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if VERBOSE:
        print("Brute force evaluation complete.")
    return best_candidate, best_score

def main():
    random.seed(42)
    num_cores = multiprocessing.cpu_count()
    pool_size = num_cores - 1 if num_cores > 1 else 1
    pool = multiprocessing.Pool(processes=pool_size)
    toolbox.register("map", pool.map)
    population = toolbox.population(n=70)
    NGEN = 12
    CXPB = 0.7
    MUTPB = 0.8
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
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
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = offspring
        current_best = tools.selBest(population, 1)[0]
        if current_best.fitness.values[0] > global_best.fitness.values[0]:
            global_best = toolbox.clone(current_best)
        fits = [ind.fitness.values[0] for ind in population]
        avg_fit = np.mean(fits)
        best_fit = max(fits)
        print(f"Generation {gen}: Avg fitness = {avg_fit:.2f}, Best fitness = {best_fit:.2f}")

    print("Overall Best individual (GA):", global_best)
    print("Overall Best fitness (GA):", global_best.fitness.values[0])
    best_indices = list(global_best)
    result_df = pot_df.iloc[best_indices].copy()
    result_df["incidi"] = best_indices
    result_df = result_df[["incidi", "latitude", "longitude"]]
    print("\nResult DataFrame (GA):")
    print(result_df)

    total_score, candidate_high, baseline_high_total = evaluate_with_breakdown(global_best)
    increase_high = candidate_high + baseline_high_total
    increase_high2 = increase_high - baseline_high_total
    print(f"\nTotal Baseline Fitness (summed over all zip regions): {total_baseline_fitness:.2f}")
    print(f"Total Baseline High Risk Coverage: {baseline_high_total:.2f}")
    print(f"Increase in High Risk Coverage from Baseline: {increase_high:.2f}")

    bf_candidate, bf_score = brute_force_verification()
    if bf_candidate is not None:
        print("\nBrute Force Verification:")
        print("Brute Force Best candidate:", bf_candidate, "with fitness:", bf_score)
        ga_score = global_best.fitness.values[0]
        if abs(bf_score - ga_score) < 1e-6:
            print("Verification successful: GA solution matches the brute force solution.")
        else:
            print(f"Verification warning: GA solution fitness ({ga_score}) differs from brute force fitness ({bf_score}).")
    else:
        print("\nBrute Force Verification skipped due to large combination count.")

    pool.close()
    pool.join()
    return global_best, result_df

if __name__ == "__main__":
    best_solution, result_df = main()
