import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    """
    Reads in the CSV file and returns a list of dictionaries.
    """
    with open(filepath, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [dict(row) for row in reader]
    return data


def calc_features(row):
    """
    Converts a dictionary representing a country into a feature vector.
    """
    keys = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
            'life_expec', 'total_fer', 'gdpp']
    return np.array([float(row[key]) for key in keys], dtype=np.float64)


def normalize_features(features):
    """
    Normalizes the feature vectors using mean and standard deviation normalization.
    """
    features_array = np.array(features)
    means = np.mean(features_array, axis=0)
    stds = np.std(features_array, axis=0)

    normalized = (features_array - means) / stds
    return [np.array(vec, dtype=np.float64) for vec in normalized]


def hac(features):
    """
    Performs Hierarchical Agglomerative Clustering using complete linkage.
    """
    n = len(features)
    Z = np.zeros((n - 1, 4))
    clusters = {i: [i] for i in range(n)}
    
    # Create the initial distance matrix
    distance_matrix = np.full((n + n - 1, n + n - 1), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(features[i] - features[j])

    next_cluster = n  # Index for new clusters

    for step in range(n - 1):
        # Find the pair with the smallest distance
        min_dist = np.inf
        cluster1, cluster2 = -1, -1

        for i in clusters:
            for j in clusters:
                if i < j and distance_matrix[i, j] < min_dist:
                    min_dist = distance_matrix[i, j]
                    cluster1, cluster2 = i, j

        # Merge clusters
        clusters[next_cluster] = clusters.pop(cluster1) + clusters.pop(cluster2)

        # Store in Z
        Z[step] = [cluster1, cluster2, min_dist, len(clusters[next_cluster])]

        # Update distance matrix with complete linkage
        for i in clusters:
            if i != next_cluster:
                distance_matrix[i, next_cluster] = distance_matrix[next_cluster, i] = max(
                    distance_matrix[p, q] for p in clusters[i] for q in clusters[next_cluster]
                )

        next_cluster += 1

    return Z


def fig_hac(Z, names):
    """
    Plots the hierarchical clustering dendrogram and returns the figure.
    """
    fig = plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8)
    plt.tight_layout()
    return fig  # Explicitly return the figure



if __name__ == "__main__":
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    
    features_normalized = normalize_features(features)
    Z = hac(features_normalized[:20])  # Example with 20 countries
    fig_hac(Z, names[:20])
