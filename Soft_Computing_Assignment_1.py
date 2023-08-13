

import numpy as np
from sklearn import preprocessing


File_data = np.loadtxt("/content/drive/MyDrive/iris_revised.txt",dtype=float)
#to create the Normalized Matrix
min_val = np.min( File_data)
max_val = np.max( File_data)

scaled_matrix = (File_data - min_val) / (max_val - min_val)
print(scaled_matrix)


# Function to compute Euclidean distance
def euclidean_distance(p1, p2): #value1,value2
    return np.sqrt(np.sum((p1 - p2)**2))
# Function to create the similarity matrix
def create_similarity_matrix(data):
    m = data.shape[0] #As we need a SM of size mxm.
    similarity_matrix = np.zeros((m, m)) #Initialized SM with 0.

    for i in range(m):
        for j in range(m):
            similarity_matrix[i, j] = euclidean_distance(data[i], data[j])

    return similarity_matrix

# Create the similarity matrix
similarity_matrix = create_similarity_matrix(File_data)

print("Similarity Matrix:")
print(similarity_matrix)

# Function to create clusters based on average dissimilarity
def create_clusters(similarity_matrix):
    m = similarity_matrix.shape[0]
    clusters = []

    for i in range(m):
        # Calculate average dissimilarity for the i-th object
        avg_dissimilarity = np.mean(similarity_matrix[i])
        cluster_i = [i]  # Initialize a cluster with the i-th object

        # Add objects with dissimilarity less than the average to the cluster
        for j in range(m):
            if similarity_matrix[i, j] < avg_dissimilarity:
                cluster_i.append(j)

        clusters.append(cluster_i)

    return clusters

# Create clusters
clusters = create_clusters(similarity_matrix)


# Display the clusters
print("Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

### Function to create clusters based on average dissimilarity
def create_clusters(similarity_matrix):
    m = similarity_matrix.shape[0]
    clusters = []

    for i in range(m):
        # Calculate average dissimilarity for the i-th object
        avg_dissimilarity = np.mean(similarity_matrix[i])
        cluster_i = {i}  # Initialize a cluster with the i-th object

        # Add objects with dissimilarity less than the average to the cluster
        for j in range(m):
            if similarity_matrix[i, j] < avg_dissimilarity:
                cluster_i.add(j)

        clusters.append(cluster_i)

    return clusters  ###

# Function to remove subsets of clusters
def remove_subset_clusters(clusters):
    non_subset_clusters = []
    for cluster in clusters:
        is_subset = False
        for other_cluster in clusters:
            if cluster != other_cluster and cluster.issubset(other_cluster):
                is_subset = True
                break
        if not is_subset:
            non_subset_clusters.append(cluster)
    return non_subset_clusters

# Create the similarity matrix
similarity_matrix = create_similarity_matrix(File_data)

# Create clusters
clusters = create_clusters(similarity_matrix)

# Remove subset clusters
final_clusters = remove_subset_clusters(clusters)

# Display the final clusters
print("Final Clusters:")
for i, cluster in enumerate(final_clusters):
    print(f"Cluster {i + 1}: {list(cluster)}")

# Function to create the similarity matrix between clusters
def create_cluster_similarity_matrix(clusters):
    p = len(clusters)
    similarity_matrix = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            intersection = len(clusters[i].intersection(clusters[j]))
            union = len(clusters[i].union(clusters[j]))
            similarity_matrix[i, j] = intersection / union

    return similarity_matrix

# Create the similarity matrix
similarity_matrix = create_similarity_matrix(File_data)

# Create clusters
clusters = create_clusters(similarity_matrix)

# Remove subset clusters
final_clusters = remove_subset_clusters(clusters)

# Create the similarity matrix between clusters
cluster_similarity_matrix = create_cluster_similarity_matrix(final_clusters)

# Display the final clusters
print("Final Clusters:")
for i, cluster in enumerate(final_clusters):
    print(f"Cluster {i + 1}: {list(cluster)}")

# Display the cluster similarity matrix
print("\nCluster Similarity Matrix:")
print(cluster_similarity_matrix)

import random  # Import the random module

# Function to find the indices of the maximum value in the matrix C
def find_max_similarity_indices(similarity_matrix):
    max_value = np.max(similarity_matrix)
    max_indices = np.where(similarity_matrix == max_value)
    k, l = random.choice(list(zip(max_indices[0], max_indices[1])))
    return k, l

# Function to merge the two most similar clusters
def merge_clusters(clusters, k, l):
    clusters[k] = clusters[k].union(clusters[l])
    del clusters[l]
    return clusters

# Create the similarity matrix
similarity_matrix = create_similarity_matrix(File_data)

# Create clusters
clusters = create_clusters(similarity_matrix)

# Remove subset clusters
final_clusters = remove_subset_clusters(clusters)

# Create the similarity matrix between clusters
cluster_similarity_matrix = create_cluster_similarity_matrix(final_clusters)

# Find the indices of the maximum value in the similarity matrix
k, l = find_max_similarity_indices(cluster_similarity_matrix)

# Merge the two most similar clusters
final_clusters = merge_clusters(final_clusters, k, l)

# Display the final clusters after merging
print("Final Clusters:")
for i, cluster in enumerate(final_clusters):
    print(f"Cluster {i + 1}: {list(cluster)}")

#This code will output the final clusters after merging the two most similar clusters. The clusters Ck and Cl with the highest similarity value are merged into a new cluster Ckl. Note that the new merged clusters will be different each time the code is run due to the random selection of the maximum similarity indices.