import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
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
        for j in range(i+1, n):
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
    
    total_dist = sum(distancia_euclideana(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))
    
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
            b = route[(i+1) % len(route)] 
            increase = (np.linalg.norm(coords[a] - coords[next_city]) +
                        np.linalg.norm(coords[next_city] - coords[b]) -
                        np.linalg.norm(coords[a] - coords[b]))
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        
        route.insert(best_pos, next_city)
        visited.add(next_city)

    route.append(route[0])

    total_dist = sum(np.linalg.norm(coords[route[i]] - coords[route[i+1]]) for i in range(len(route)-1))
    
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
            b = route[(i+1) % len(route)]
            increase = (np.linalg.norm(coords[a] - coords[next_city]) +
                        np.linalg.norm(coords[next_city] - coords[b]) -
                        np.linalg.norm(coords[a] - coords[b]))
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        
        route.insert(best_pos, next_city)
        visited.add(next_city)
    
    route.append(route[0])
    
    total_dist = sum(np.linalg.norm(coords[route[i]] - coords[route[i+1]]) for i in range(len(route)-1))
    
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



