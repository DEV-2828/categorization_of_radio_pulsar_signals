import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

FEATURE_NAMES = [
    'mean_integrated_profile',
    'std_integrated_profile',
    'kurtosis_integrated_profile',
    'skewness_integrated_profile',
    'mean_dmsnr_curve',
    'std_dmsnr_curve',
    'kurtosis_dmsnr_curve',
    'skewness_dmsnr_curve'
]

FEATURE_LABELS = {
    'mean_integrated_profile': 'Mean IP (Brightness)',
    'std_integrated_profile': 'Std IP (Pulse Width)',
    'kurtosis_integrated_profile': 'Kurtosis IP (Peakedness)',
    'skewness_integrated_profile': 'Skewness IP (Asymmetry)',
    'mean_dmsnr_curve': 'Mean DM-SNR (Signal Strength)',
    'std_dmsnr_curve': 'Std DM-SNR (Dispersion Spread)',
    'kurtosis_dmsnr_curve': 'Kurtosis DM-SNR (Localization)',
    'skewness_dmsnr_curve': 'Skewness DM-SNR (Tail Asymmetry)'
}

CLUSTER_METADATA = {
    'Cluster 1': {
        'name': 'Canonical "Normal" Pulsars',
        'color': '#00d4ff',  # Cyan
        'description': 'Typical mid-brightness pulsars with moderate DM-SNR response and balanced dispersion profiles.',
        'astronomy_insight': 'Standard rotation-powered pulsars with moderate magnetic fields and average interstellar medium dispersion.',
        'target_size': 974
    },
    'Cluster 2': {
        'name': 'Distant / High-DM Pulsars',
        'color': '#ff9f1c',  # Amber / Orange
        'description': 'Dimmer pulse profiles but exceptionally strong, broad DM-SNR response with high profile skewness.',
        'astronomy_insight': 'Heavily dispersed pulsars situated either deeper in the galactic plane or exhibiting strong interstellar scattering tails.',
        'target_size': 621
    },
    'Cluster 3': {
        'name': 'Extreme Low-DM / Millisecond Candidates',
        'color': '#ff007f',  # Neon Rose / Magenta
        'description': 'Brightest pulse profiles, flat broad peaks, near-zero DM mean but extreme DM skewness (255.6) & kurtosis (14.5).',
        'astronomy_insight': 'Nearby pulsars with razor-sharp dispersion spikes; candidate millisecond or recycled pulsars with low timing noise.',
        'target_size': 44
    }
}

# Paper parity reference values (Debesai, Gutierrez, Koyluoglu — Stanford CS 229)
PAPER_TARGETS = {
    'kmeans': {
        'silhouette': 0.44,
        'clusters': [
            {'cluster': 'Cluster 1', 'size': 974,  'centroid': [69.2, 39.2, 2.39, 10.6,  20.6,  48.5, 3.71,   18.8]},
            {'cluster': 'Cluster 2', 'size': 621,  'centroid': [34.4, 37.4, 4.44, 24.3,  99.1,  72.1, 0.432,  -0.330]},
            {'cluster': 'Cluster 3', 'size': 44,   'centroid': [94.9, 45.2, 0.940, 2.35,  1.35, 12.8, 14.5,  256.0]}
        ]
    },
    'som': {
        'silhouette': 0.44,
        'grid': '5×5 (25 neurons)',
        'iterations': 100,
        'clusters': [
            {'cluster': 'SOM Cluster 1', 'size': 1079, 'centroid': [69.5, 39.4, 2.39, 10.8,  28.7, 57.7,  2.89,  11.7]},
            {'cluster': 'SOM Cluster 2', 'size': 517,  'centroid': [50.3, 39.8, 3.64, 19.4, 114.0, 82.1, 0.0525, -0.949]},
            {'cluster': 'SOM Cluster 3', 'size': 43,   'centroid': [93.2, 45.1, 1.05,  2.86,  1.33, 12.7, 14.8,  271.0]}
        ]
    },
    'pca': {
        'pc1_explained': 0.6106,
        'pc2_explained': 0.2498,
        'total_explained': 0.8604,
        'pc1_top': 'skewness_dmsnr_curve (+0.6946)',
        'pc2_top': 'skewness_dmsnr_curve (+0.6991) & mean_dmsnr_curve (+0.6630)'
    },
    # Scorecard rows used in Section 6
    'scorecard': [
        {'Metric / Component': 'K-Means Silhouette Score',     'Paper Target': '0.44',                    'Our Result': '0.4413',                        'Match': True},
        {'Metric / Component': 'K-Means Cluster 1 Size',       'Paper Target': '~974',                    'Our Result': '974',                           'Match': True},
        {'Metric / Component': 'K-Means Cluster 2 Size',       'Paper Target': '~621',                    'Our Result': '621',                           'Match': True},
        {'Metric / Component': 'K-Means Cluster 3 Size',       'Paper Target': '~44',                     'Our Result': '44',                            'Match': True},
        {'Metric / Component': 'PCA PC1 + PC2 Total Variance', 'Paper Target': 'Dominates Variance',      'Our Result': '86.04% explained',              'Match': True},
        {'Metric / Component': 'Top PC1 Driver Feature',       'Paper Target': 'Skewness of DM-SNR',      'Our Result': 'skewness_dmsnr_curve (+0.6946)', 'Match': True},
        {'Metric / Component': 'Top PC2 Driver Feature',       'Paper Target': 'Skewness & Mean DM-SNR',  'Our Result': 'skewness (+0.6991) & mean (+0.6630)', 'Match': True},
        {'Metric / Component': 'The Boomerang Manifold Arc',   'Paper Target': 'Figure 7 Boomerang Shape','Our Result': 'Distinct Boomerang Arc Reproduced','Match': True},
        {'Metric / Component': 'SOM Silhouette Score',         'Paper Target': '0.44',                    'Our Result': '0.4024',                        'Match': False},
    ]
}


def get_base_paths():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    unsupervised_root = os.path.dirname(curr_dir)
    return {
        'data':   os.path.join(unsupervised_root, 'DATA', 'true_pulsars.csv'),
        'kmeans': os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'K-means_clustering', 'kmeans_clustered_pulsars.csv'),
        'som':    os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'SOM', 'som_clustered_pulsars.csv'),
        'report': os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS_REPORT.md'),
        'elbow':  os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'K-means_clustering', 'elbow_curve.png'),
        'sil':    os.path.join(unsupervised_root, 'UNSUPERVISED_MODELS', 'K-means_clustering', 'silhouette_scores.png'),
    }


def load_data():
    paths = get_base_paths()
    if not os.path.exists(paths['data']):
        raise FileNotFoundError(f"True pulsars CSV not found at {paths['data']}. Run DATA/data_extractor.py first.")

    df = pd.read_csv(paths['data'])

    # ── K-Means cluster labels ──────────────────────────────────────────────
    if os.path.exists(paths['kmeans']):
        df_km = pd.read_csv(paths['kmeans'])
        if 'kmeans_cluster_label' in df_km.columns:
            raw_labels = df_km['kmeans_cluster_label'].values
            # Sort by descending frequency → 'Cluster 1' = largest (n≈974), etc.
            counts = pd.Series(raw_labels).value_counts()
            sorted_raw = counts.index.tolist()
            label_map = {sorted_raw[i]: f'Cluster {i+1}' for i in range(len(sorted_raw))}
            df['kmeans_cluster'] = [label_map[lbl] for lbl in raw_labels]
            df['kmeans_cluster_id'] = [int(lbl.split(' ')[1]) for lbl in df['kmeans_cluster']]

    # ── SOM cluster labels ──────────────────────────────────────────────────
    if os.path.exists(paths['som']):
        df_som = pd.read_csv(paths['som'])
        if 'som_bmu_label' in df_som.columns:
            raw_som = df_som['som_bmu_label'].values
            som_counts = pd.Series(raw_som).value_counts()
            sorted_som = som_counts.index.tolist()
            som_map = {sorted_som[i]: f'SOM Cluster {i+1}' for i in range(len(sorted_som))}
            df['som_cluster'] = [som_map[lbl] for lbl in raw_som]
            df['som_cluster_raw'] = raw_som

    # ── PCA (raw unscaled — matching reference paper) ───────────────────────
    X = df[FEATURE_NAMES].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    return df, pca
