"""
Authour: @Raj Gandhi

The code takes in point cloud data and runs RANSAC clustering alorithm on it 
and segments out the point cloud to inler and outlier sets.

thershold = distance between two point groups.
iterations = number of iterations to run

"""

import random
import numpy as np
import time

def ransac_plane(xyz, threshold=0.055, iterations=1000):
    inliers=[]
    n_points=len(xyz)
    i=1
    while i<iterations:
        idx_samples = random.sample(range(n_points), 3)
        pts = xyz[idx_samples]
        vecA = pts[1] - pts[0]
        vecB = pts[2] - pts[0]
        normal = np.cross(vecA, vecB) 
        a,b,c = abs(normal / np.linalg.norm(normal))
        d=-np.sum(normal*pts[1])
        distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        if len(idx_candidates) > len(inliers):
            equation = [a,b,c,d]
            inliers = idx_candidates      
        i+=1
    return equation, inliers

if __name__ == '__main__':
    #import point cloud
    pcd = np.loadtxt("point_cloud_points.txt")

    #run RANSAC
    eq,idx_inliers=ransac_plane(pcd)
    print(eq)
    inliers=pcd[idx_inliers]

    mask = np.ones(len(pcd), dtype=bool)
    mask[idx_inliers] = False
    outliers=pcd[mask]
    print(len(outliers))

    np.savetxt("C:/temp/inliers.txt", inliers)
    np.savetxt("C:/temp/outliers.txt", outliers)