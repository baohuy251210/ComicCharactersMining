from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering as Cluster
import matplotlib.colors as mcolors
from math import sqrt

TAB_COLORS = list(mcolors.TABLEAU_COLORS)

#Plots a scatter plot of each point colored by cluster
def plot_clusters(ax, labels, vals):
    colors = [TAB_COLORS[i] for i in labels]
    ax.scatter(vals[:,0], vals[:,1], c=colors)

#Returns the euclidean distance between pts a and b
def dist(a, b):
    x = np.array(a)
    y = np.array(b)
    return np.linalg.norm(x - y)

#Returns the center that minimizes the distance between a point and that center
def phi(centers, pt):
    min_d = np.Inf
    arg_min = None
    for center in centers:
        d = dist(center, pt)
        if d < min_d:
            min_d = d
            arg_min = center
    return arg_min

#Finds n single-link hierarchical clusters for pts
def single_link_cluster(pts, n_clusters):
    c = Cluster(n_clusters=n_clusters, linkage='single')
    return c.fit(pts).labels_

#Finds n complete-link hierarchical clusters for pts
def complete_lin_cluster(pts, n_clusters):
    c = Cluster(n_clusters=n_clusters, linkage='complete')
    return c.fit(pts).labels_

#Returns an index arr of the center nearest to each pt in pts
def assign_to_centers(pts, centers):
    return [centers.index(phi(centers, pt)) for pt in pts]

#Returns the centers and the assigned labels found for the pts
def gonzalez_clustering(pts, n_centers):
    centers = [None for _ in range(n_centers)]
    pts = list(map(tuple, pts))
    centers[0] = pts[0]
    for i in range(1, n_centers):
        max_d = np.NINF
        arg_max = None
        for pt in pts:
            d = dist(pt, phi(centers[0:i], pt))
            if d > max_d:
                max_d = d
                arg_max = pt
        centers[i] = arg_max
    return (centers, assign_to_centers(pts, centers))

#Returns the centers and the assigned labels found for the pts
def k_means_pp_clustering(pts, n_centers):
    centers = [None for _ in range(n_centers)]
    pts = list(map(tuple, pts))
    centers[0] = pts[0]
    for i in range(1, n_centers):
        dists = np.array([dist(p, phi(centers[0:i], p))**2 for p in pts])
        probs = dists / sum(dists)
        centers[i] = pts[np.random.choice(len(pts), p=probs)]
    return (centers, assign_to_centers(pts, centers))

#Returns the component-wise average for a collection of pts
def pt_average(pts):
    return tuple(np.mean(pts, axis=0))

#Performs lloyds algorithm on a given collection of pts and centers
def lloyds_alg(pts, centers):
    tol = 0.1
    change = tol + 1
    result = centers
    pts = list(map(tuple, pts))
    while change > tol:
        assignments = assign_to_centers(pts, result)
        new_centers = [None for _ in range(len(centers))]
        for i in range(len(centers)):
            assigned_pts = [pts[j] for j in range(len(pts)) if assignments[j] == i]
            new_centers[i] = pt_average(assigned_pts)
        change = sum(dist(a, b) for a, b in zip(result, new_centers))
        result = new_centers
    return (result, assign_to_centers(pts, result))

#Runs Lloyd's algorithm to refine the cluster centers
#found from Gonzales algorithm
def gonzalloyds(pts, n_centers):
    centers, _ = gonzalez_clustering(pts, n_centers)
    return lloyds_alg(pts, centers) 

#k-center cost for a collection of points and centers
def k_center_cost(centers, pts):
    return max(dist(pt, phi(centers, pt)) for pt in pts)

#k-means cost for a collection of points and centers
def k_means_cost(centers, pts):
    return sqrt((sum(dist(pt, phi(centers, pt))**2 for pt in pts))/len(pts))

#Finds the minimum distance between two points in a pair of clusters
def min_cluster_dist(a, b):
    min_dist = np.Inf
    for a_pt in a:
        for b_pt in b:
            min_dist = min(dist(a_pt, b_pt), min_dist)
    return min_dist

#Returns the average minimum distance between two points in
#each pair of clusters
def avg_min_cluster_dist(labels, pts):
    # Organize points by cluster assignment
    clusters = {i : [] for i in set(labels)}
    for index, pt in zip(labels, pts):
        clusters[index].append(pt)

    # Find min distances between clusters
    min_distances = []
    for a, b in combinations(clusters.keys(), 2):
        min_dist = min_cluster_dist(clusters[a], clusters[b])
        min_distances.append(min_dist)
    #Compute means for each center and return    
    return np.mean(min_distances)