import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist, squareform
import ast

df = pd.read_csv("tsp_dataset.csv")


def coordenadas_instancia(row):
    coords_list = ast.literal_eval(row['city_coordinates'])
    coords = np.array(coords_list, dtype=float)
    return coords


def vizinho_mais_proximo(coords):
    n = len(coords)
    visited = [False] * n
    route = [0]
    visited[0] = True
    total_dist = 0.0

    for _ in range(n - 1):
        last = route[-1]
        dists = np.linalg.norm(coords - coords[last], axis=1)
        dists = np.where(visited, np.inf, dists)
        next_city = np.argmin(dists)

        route.append(next_city)
        visited[next_city] = True
        total_dist += dists[next_city]

    total_dist += np.linalg.norm(coords[route[-1]] - coords[route[0]])

    return total_dist


def distancia_euclideana(a, b):
    return np.linalg.norm(a - b)


def christofides(coords):
    n = len(coords)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            dist = distancia_euclideana(coords[i], coords[j])
            G.add_edge(i, j, weight=dist)

    AGM = nx.minimum_spanning_tree(G)
    odd_nodes = [v for v, d in AGM.degree() if d % 2 == 1]

    odd_graph = nx.Graph()
    for u, v in combinations(odd_nodes, 2):
        odd_graph.add_edge(u, v, weight=distancia_euclideana(coords[u], coords[v]))

    matching = nx.algorithms.matching.min_weight_matching(odd_graph)
    multi_graph = nx.MultiGraph(AGM)
    multi_graph.add_edges_from(matching)
    odd_left = [v for v, d in multi_graph.degree() if d % 2 == 1]

    while odd_left:
        u = odd_left.pop()
        v = min(odd_left, key=lambda x: distancia_euclideana(coords[u], coords[x]))
        multi_graph.add_edge(u, v, weight=distancia_euclideana(coords[u], coords[v]))
        odd_left.remove(v)

    euler_circuit = list(nx.eulerian_circuit(multi_graph))
    visited = set()
    route = []
    for u, v in euler_circuit:
        if u not in visited:
            route.append(u)
            visited.add(u)
        if v not in visited:
            route.append(v)
            visited.add(v)

    route.append(route[0])

    total_dist = sum(distancia_euclideana(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1))

    return total_dist


def nearest_insertion(coords):
    n = len(coords)

    route = [0, 1]
    visited = set(route)

    while len(route) < n:
        min_dist = np.inf
        next_city = None
        insert_pos = None

        for city in range(n):
            if city in visited:
                continue
            dists_to_tour = [np.linalg.norm(coords[city] - coords[r]) for r in route]
            closest_dist = min(dists_to_tour)
            if closest_dist < min_dist:
                min_dist = closest_dist
                next_city = city

        best_increase = np.inf
        for i in range(len(route)):
            a = route[i]
            b = route[(i + 1) % len(route)]
            increase = (np.linalg.norm(coords[a] - coords[next_city]) +
                        np.linalg.norm(coords[next_city] - coords[b]) -
                        np.linalg.norm(coords[a] - coords[b]))
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        route.insert(best_pos, next_city)
        visited.add(next_city)

    route.append(route[0])

    total_dist = sum(np.linalg.norm(coords[route[i]] - coords[route[i + 1]]) for i in range(len(route) - 1))

    return total_dist


import numpy as np
import random


def random_insertion(coords):
    n = len(coords)

    route = random.sample(range(n), 2)
    visited = set(route)

    while len(route) < n:
        remaining = [c for c in range(n) if c not in visited]
        next_city = random.choice(remaining)

        best_increase = np.inf
        best_pos = None
        for i in range(len(route)):
            a = route[i]
            b = route[(i + 1) % len(route)]
            increase = (np.linalg.norm(coords[a] - coords[next_city]) +
                        np.linalg.norm(coords[next_city] - coords[b]) -
                        np.linalg.norm(coords[a] - coords[b]))
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        route.insert(best_pos, next_city)
        visited.add(next_city)

    route.append(route[0])

    total_dist = sum(np.linalg.norm(coords[route[i]] - coords[route[i + 1]]) for i in range(len(route) - 1))

    return total_dist


coords0 = coordenadas_instancia(df.loc[1964])

distanciavmp = vizinho_mais_proximo(coords0)
print("Distância 1:", distanciavmp)
distanciac = christofides(coords0)
print("Distância 2:", distanciac)
distanciani = nearest_insertion(coords0)
print("Distância 3:", distanciani)
distanciari = random_insertion(coords0)
print("Distância 4:", distanciari)

sample_df = df.sample(10, random_state=42)
results = []
for idx, row in sample_df.iterrows():
    coords = coordenadas_instancia(row)
    dist_vmp = vizinho_mais_proximo(coords)
    dist_chris = christofides(coords)
    dist_ni = nearest_insertion(coords)
    dist_ri = random_insertion(coords)

    dist_dict = {'vizinho_mais_proximo': dist_vmp,
                 "christofides": dist_chris,
                 'nearest_insertion': dist_ni,
                 'random_insertion': dist_ri}
    best = min(dist_dict, key=dist_dict.get)

    results.append({
        'instance_id': row['instance_id'],
        'num_cities': row['num_cities'],
        'best_heuristic': best
    })

df_results = pd.DataFrame(results)
df_results.to_csv("tsp_heuristics_results.csv", index=False)
print("Resultados guardados em 'tsp_heuristics_results.csv'")


def extractfeatures(coords, distance_matrix=None):
    cities = np.array(coords, dtype=np.float64)
    n = len(cities)
    features = {}
    features['num_cities'] = n
    features['hull_area_ratio'] = _compute_hull_area_ratio(cities)
    features['hopkins_statistic'] = _compute_hopkins_statistic(cities)
    features['convexity_defect'] = _compute_convexity_defect(cities)
    if distance_matrix is None:
        distance_matrix = squareform(pdist(cities))
    features['mst_ratio'] = _compute_mst_ratio(cities, distance_matrix)
    features['hull_points_ratio'] = _compute_hull_points_ratio(cities)
    features['clark_evans_index'] = _compute_clark_evans_index(cities, distance_matrix)
    dist_features = _compute_distance_features(distance_matrix)
    features['distance_ratio'] = dist_features['ratio']
    features['distance_std'] = dist_features['std']
    features['distance_mean'] = dist_features['mean']
    features['nn_mean'] = _compute_nn_mean(distance_matrix)
    features['bbox_area'] = _compute_bbox_area(cities)
    features['xy_correlation'] = _compute_xy_correlation(cities)
    features['mst_length'] = _compute_mst_length(distance_matrix)
    features['coord_spread_ratio'] = _compute_coord_spread_ratio(cities)

    return features


def _compute_hull_area_ratio(cities):
    n = len(cities)
    if n < 3:
        return 1.0
    try:
        hull = ConvexHull(cities)
        hull_area = hull.volume
        bbox_area = _compute_bbox_area(cities)
        return hull_area / bbox_area if bbox_area > 0 else 1.0
    except:
        return 1.0


def _compute_hopkins_statistic(cities, n_samples=None):
    n = len(cities)
    if n < 10:
        return 0.5
    if n_samples is None:
        n_samples = min(100, n // 2)

    x_min, x_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    y_min, y_max = np.min(cities[:, 1]), np.max(cities[:, 1])

    random_points = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(n_samples, 2)
    )

    sample_indices = np.random.choice(n, size=n_samples, replace=False)
    sample_points = cities[sample_indices]

    tree = KDTree(cities)

    u_distances, _ = tree.query(random_points, k=1)
    u_sum = np.sum(u_distances)

    w_distances, _ = tree.query(sample_points, k=2)
    w_sum = np.sum(w_distances[:, 1])

    if u_sum + w_sum > 0:
        return u_sum / (u_sum + w_sum)
    return 0.5


def _compute_convexity_defect(cities):
    n = len(cities)
    if n < 4:
        return 0.0
    try:
        hull = ConvexHull(cities)
        hull_points = cities[hull.vertices]
        hull_set = set([tuple(p) for p in hull_points])
        interior_points = [p for p in cities if tuple(p) not in hull_set]
        if not interior_points:
            return 0.0

        interior_points = np.array(interior_points)

        # Distance to nearest hull point
        hull_tree = KDTree(hull_points)
        distances, _ = hull_tree.query(interior_points)

        return np.mean(distances)
    except:
        return 0.0


def _compute_mst_ratio(cities, distance_matrix):
    n = len(cities)
    if n < 2:
        return 0.0
    mst_length = _compute_mst_length(distance_matrix)
    upper_idx = np.triu_indices(n, 1)
    avg_distance = np.mean(distance_matrix[upper_idx]) if len(upper_idx[0]) > 0 else 0
    if avg_distance > 0:
        return mst_length / (n * avg_distance)
    return 0.0


def _compute_hull_points_ratio(cities):
    n = len(cities)
    if n < 3:
        return 1.0
    try:
        hull = ConvexHull(cities)
        return len(hull.vertices) / n
    except:
        return 1.0


def _compute_clark_evans_index(cities, distance_matrix):
    n = len(cities)
    if n < 2:
        return 1.0

    masked_dist = distance_matrix.copy().astype(float)
    np.fill_diagonal(masked_dist, np.nan)

    nn_dists = np.nanmin(masked_dist, axis=1)
    if np.any(~np.isfinite(nn_dists)):
        return 1.0

    ro_observed = np.mean(nn_dists)

    bbox_area = _compute_bbox_area(cities)
    if not np.isfinite(bbox_area) or bbox_area <= 0:
        return 1.0

    density = n / bbox_area
    if not np.isfinite(density) or density <= 0:
        return 1.0

    re_expected = 0.5 / np.sqrt(density)
    if not np.isfinite(re_expected) or re_expected <= 0:
        return 1.0

    return float(ro_observed / re_expected)

def _compute_distance_features(distance_matrix):
    n = len(distance_matrix)
    upper_idx = np.triu_indices(n, 1)
    distances = distance_matrix[upper_idx]

    if len(distances) == 0:
        return {'mean': 0.0, 'std': 0.0, 'ratio': 0.0}

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    ratio = max_dist / min_dist if min_dist > 0 else 0.0

    return {'mean': mean_dist, 'std': std_dist, 'ratio': ratio}


def _compute_nn_mean(distance_matrix):
    n = distance_matrix.shape[0]
    if n < 2:
        return 0.0
    masked_dist = distance_matrix.copy().astype(float)
    np.fill_diagonal(masked_dist, np.nan)

    nn_dists = np.nanmin(masked_dist, axis=1)
    if np.any(~np.isfinite(nn_dists)):
        return 0.0

    return float(np.mean(nn_dists))

def _compute_bbox_area(cities):
    x_min, x_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    y_min, y_max = np.min(cities[:, 1]), np.max(cities[:, 1])
    return (x_max - x_min) * (y_max - y_min)

def _compute_xy_correlation(cities):
    n = len(cities)
    if n < 2:
        return 0.0

    x = cities[:, 0]
    y = cities[:, 1]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    correlation = np.corrcoef(x, y)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def _compute_mst_length(distance_matrix):
    n = len(distance_matrix)
    if n < 2:
        return 0.0

    visited = [False] * n
    min_edge = [np.inf] * n
    min_edge[0] = 0
    total_length = 0.0
    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i

        visited[u] = True
        total_length += min_edge[u]

        for v in range(n):
            if not visited[v] and distance_matrix[u][v] < min_edge[v]:
                min_edge[v] = distance_matrix[u][v]

    return total_length


def _compute_coord_spread_ratio(cities):
    x_std = np.std(cities[:, 0])
    y_std = np.std(cities[:, 1])

    if y_std > 0:
        return x_std / y_std
    return 1.0 if x_std == 0 else np.inf



def extract_X_y_from_instance(row):

    coords = coordenadas_instancia(row)

    dist_vmp = vizinho_mais_proximo(coords)
    dist_chris = christofides(coords)
    dist_ni = nearest_insertion(coords)
    dist_ri = random_insertion(coords)

    dist_dict = {
        'vizinho_mais_proximo': dist_vmp,
        'christofides': dist_chris,
        'nearest_insertion': dist_ni,
        'random_insertion': dist_ri
    }

    y = min(dist_dict, key=dist_dict.get)
    distance_matrix = squareform(pdist(coords))
    X = extractfeatures(coords, distance_matrix)

    return X, y

def build_feature_dataset(df):
    rows = []
    for _, row in df.iterrows():
        X, y = extract_X_y_from_instance(row)

        row_out = {
            'instance_id': row['instance_id'],
            'best_heuristic': y
        }
        row_out.update(X)
        rows.append(row_out)

    return pd.DataFrame(rows)

sample_df = df.head(2226)

df_ml = build_feature_dataset(sample_df)

df_ml.to_csv("tsp_features_heuristics_results.csv", index=False)

print("CSV guardado em 'tsp_features_heuristics_results.csv'")

