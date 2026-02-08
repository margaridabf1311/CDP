import pandas as pd
import numpy as np

df = pd.read_csv("tsp_instances_dataset.csv")

def coordenadas_instancia(row):
    coords = []
    i = 1
    while f"City_{i}_X" in row:
        x = row[f"City_{i}_X"]
        y = row[f"City_{i}_Y"]

        if pd.isna(x) or pd.isna(y):
            break

        coords.append([float(x), float(y)])
        i += 1
    return np.array(coords)


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

coords0 = coordenadas_instancia(df.loc[0])

distanciavmp = vizinho_mais_proximo(coords0)
print("Distância 1:", distanciavmp)

