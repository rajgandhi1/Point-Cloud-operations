"""
Authour: @Raj Gandhi

The code takes in point cloud data and runs DBSCAN clustering alorithm on it 
and segments out the point cloud whith the biggest number of points in it.

eps = distance threshold.
min_samples = number of points in a group.

uses sklearns DBSCAN implementation.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN

#load in point cloud that is in format of x, y, z point format.
pcd = np.loadtxt("point_cloud_points.txt")

db = DBSCAN(eps=0.05, min_samples=10).fit(pcd)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
best_candidate=int(np.unique(labels)[np.where(n_clusters_== np.max(n_clusters_))[0]])
segments_t=pcd.take(list(np.where(labels== best_candidate)[0]), axis=0)
print(len(segments_t))
print(pcd.shape)
print(segments_t.shape)
np.savetxt("point_cloud_segmentes.txt", segments_t)