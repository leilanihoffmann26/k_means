import numpy as np
import pandas as pd

file_path = "/Users/leilanihoffmann/test.csv"
data = pd.read_csv(file_path)
X = data.to_numpy()

def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, iters=10):
    centroids = initialize_centroids(X, k)
    
    for i in range(iters):
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)
               
    return labels, centroids



k = 3
labels, centroids = kmeans(X, k)

print("Cluster labels:", labels)
print("Centroids:", centroids)


