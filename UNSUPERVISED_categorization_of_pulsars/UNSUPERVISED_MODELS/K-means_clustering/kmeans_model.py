import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import collections
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

def run_kmeans():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    unsupervised_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(unsupervised_root, 'DATA', 'true_pulsars.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("1. Loading Data...")
    df = pd.read_csv(data_path)
    X = df.values
    
    # -----------------------------------------------------------------------
    # NOTE ON SCALING:
    # The reference paper (Debesai, Gutierrez, Koyluoglu) ran K-Means on the
    # RAW (unscaled) feature data. This is confirmed by the centroids reported
    # in Table 2 of the paper, which are in the original feature space (e.g.,
    # [34.4, 37.4, 4.44, 24.3, 99.1, 72.1, 0.432, −0.330]).
    # The Silhouette Coefficient of 0.44 was also computed on raw data.
    #
    # In this dataset, the DM-SNR features (especially Mean and Skewness)
    # have naturally larger magnitudes and variance, so they dominate the
    # Euclidean distance in the unscaled space. This is actually desirable
    # here because those features are the most discriminative for pulsar
    # sub-categorization (as the PCA analysis confirms).
    # -----------------------------------------------------------------------
    
    print("\n2. Determining optimal k (Iterating k=2 to 7)...")
    k_values = range(2, 8)
    wcss = []
    silhouette_scores_list = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X, labels)
        silhouette_scores_list.append(score)
        
        print(f"  -> k={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
        
    # Plotting Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Curve for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    elbow_path = os.path.join(script_dir, 'elbow_curve.png')
    plt.savefig(elbow_path)
    plt.close()
    
    # Plotting Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores_list, marker='o', linestyle='-', color='orange')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    silhouette_path = os.path.join(script_dir, 'silhouette_scores.png')
    plt.savefig(silhouette_path)
    plt.close()
    
    print(f"  -> Saved 'elbow_curve.png' and 'silhouette_scores.png' to {script_dir}")
    
    print("\n3. Executing K-Means with k=3...")
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_3 = kmeans_3.fit_predict(X)
    
    print("\n4. Benchmark Results (Targeting k=3):")
    final_score = silhouette_score(X, labels_3)
    print(f"  -> Silhouette Score: {final_score:.4f} (Expected target: ~0.44)")
    
    # Print centroids for verification against paper's Table 2
    print("\n  -> Cluster Centroids (raw feature space):")
    sorted_indices = sorted(range(3), key=lambda i: sum(labels_3 == i), reverse=True)
    for rank, ci in enumerate(sorted_indices, 1):
        centroid = kmeans_3.cluster_centers_[ci]
        count = sum(labels_3 == ci)
        centroid_str = ', '.join([f"{v:.1f}" for v in centroid])
        print(f"     Cluster {rank}: [{centroid_str}] — {count} samples")
    
    print("\n  -> Cluster Distribution:")
    cluster_counts = collections.Counter(labels_3)
    
    # Sort by counts descending to match targets (Target 1: ~974, Target 2: ~621, Target 3: ~44)
    # The cluster IDs from K-Means are arbitrary, so we sort by size to map to the implementation plan's clusters
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    targets = [974, 621, 44]
    for i, (cluster_id, count) in enumerate(sorted_clusters, 1):
        target = targets[i - 1] if i <= len(targets) else '?'
        print(f"     Cluster {i} (Label {cluster_id}): {count} samples  [Paper target: ~{target}]")
        
    # Add labels back to dataset and save
    df['kmeans_cluster_label'] = labels_3
    output_csv = os.path.join(script_dir, 'kmeans_clustered_pulsars.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved clustered dataset to {output_csv}")

if __name__ == '__main__':
    run_kmeans()
