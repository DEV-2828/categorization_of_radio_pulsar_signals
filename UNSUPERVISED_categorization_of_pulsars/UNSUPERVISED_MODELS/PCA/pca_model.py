import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def run_pca():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    unsupervised_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(unsupervised_root, 'DATA', 'true_pulsars.csv')
    
    # Paths to previously saved clustered datasets
    kmeans_path = os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'K-means_clustering', 'kmeans_clustered_pulsars.csv')
    som_path = os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'SOM', 'som_clustered_pulsars.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("1. Loading Data...")
    df = pd.read_csv(data_path)
    X = df.values
    
    # Try loading clustering labels
    kmeans_labels = None
    if os.path.exists(kmeans_path):
        df_kmeans = pd.read_csv(kmeans_path)
        if 'kmeans_cluster_label' in df_kmeans.columns:
            kmeans_labels = df_kmeans['kmeans_cluster_label'].values
            
    som_labels = None
    if os.path.exists(som_path):
        df_som = pd.read_csv(som_path)
        if 'som_bmu_label' in df_som.columns:
            som_labels = df_som['som_bmu_label'].values
            
    # NOTE: The reference paper computed PCA on the RAW, unscaled data.
    # This is why their top contributors were DM-SNR Skewness and Mean,
    # as these features have naturally massive variances that dominate PCA.
    # We remove StandardScaler to exactly replicate Table 1 in the paper.
    
    print("\n2. Computing Principal Components on RAW (unscaled) data...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"  -> Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    
    # 3. Feature Correlation Analysis
    feature_names = [
        'mean_integrated_profile', 'std_integrated_profile',
        'kurtosis_integrated_profile', 'skewness_integrated_profile',
        'mean_dmsnr_curve', 'std_dmsnr_curve',
        'kurtosis_dmsnr_curve', 'skewness_dmsnr_curve'
    ]
    
    components = pca.components_
    print("\n3. Feature Correlation Analysis (PCA Loadings):")
    print("-" * 65)
    print(f"{'Feature':<30} | {'PC1 Loading':<14} | {'PC2 Loading'}")
    print("-" * 65)
    for i, feature in enumerate(feature_names):
        print(f"{feature:<30} | {components[0, i]:>14.4e} | {components[1, i]:>11.4e}")
    
    # Identify top contributors to PC1 and PC2
    pc1_top = feature_names[np.argmax(np.abs(components[0]))]
    pc2_top = feature_names[np.argmax(np.abs(components[1]))]
    print(f"\n  -> Top contributor to PC1: {pc1_top}")
    print(f"  -> Top contributor to PC2: {pc2_top}")
        
    print("\n4. Generating 'Boomerang' Plots...")
    
    # Plot 1: Base Boomerang Plot
    plt.figure(figsize=(9, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=15, color='#444444')
    plt.title('PCA: The Boomerang Manifold (Unclustered)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    base_plot = os.path.join(script_dir, 'pca_boomerang_base.png')
    plt.savefig(base_plot)
    plt.close()
    
    # Plot 2: K-Means Boomerang Plot
    if kmeans_labels is not None:
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7, s=20)
        plt.title('PCA Manifold (Color-coded by K-Means Clusters)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="K-Means\nClusters")
        plt.grid(True, linestyle='--', alpha=0.6)
        kmeans_plot = os.path.join(script_dir, 'pca_boomerang_kmeans.png')
        plt.savefig(kmeans_plot)
        plt.close()
        
    # Plot 3: SOM Boomerang Plot
    if som_labels is not None:
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=som_labels, cmap='tab20', alpha=0.7, s=20)
        plt.title('PCA Manifold (Color-coded by SOM BMUs)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='SOM BMU Index')
        plt.grid(True, linestyle='--', alpha=0.6)
        som_plot = os.path.join(script_dir, 'pca_boomerang_som.png')
        plt.savefig(som_plot)
        plt.close()
        
    print(f"  -> Saved all 3 scatter plots successfully to {script_dir}")

if __name__ == '__main__':
    run_pca()
