# Implementation Plan: Unsupervised Learning Pipeline (Pulsar Sub-Categorization)

## 1. Project Context & Objective
While the supervised learning pipeline aims to classify signals as pulsar (1) or noise (0), the **unsupervised learning pipeline** is strictly concerned with exploring the **1,639 true pulsars**. 

**Objective:** To determine if these true pulsars naturally group into distinct astrophysical sub-categories based on their 8 statistical features, using clustering and manifold projection techniques.

---

## 2. Data Preparation
1. **Filtering:** Isolate only the rows where the target label is `1` (true pulsars). The resulting dataset should have exactly $N = 1,639$ rows and 8 features.
2. **Standardization:** Unsupervised distance-based algorithms are highly sensitive to feature magnitudes. You **must** fit a `StandardScaler` on these 1,639 examples to normalize all 8 features before applying any clustering algorithm.

---

## 3. Model 1: K-Means Clustering (Baseline)
**Goal:** Partition the pulsars into $k$ distinct, non-overlapping clusters.

### Implementation Steps:
1. **Determining $k$:**
   - Run K-Means iteratively for $k \in \{2, 3, 4, 5, 6, 7\}$.
   - Plot the **Elbow Curve** (Within-Cluster Sum of Squares vs. $k$).
   - Plot the **Silhouette Score vs. $k$**.
   - *Expected Outcome:* The metrics should mathematically justify $k=3$ as the optimal number of clusters.
2. **Execution:** Run K-Means with $k=3$.
3. **Target Benchmarks:**
   - **Silhouette Score:** Aim for `~0.44`.
     - *How to measure:* Use `sklearn.metrics.silhouette_score(X_scaled, cluster_labels)` to compute it via code. Mathematically, it calculates the mean Silhouette Coefficient across all samples. For a single sample, the formula is `s = (b - a) / max(a, b)`, where `a` is the mean distance between the sample and all other points in the same cluster (intra-cluster distance), and `b` is the mean distance between the sample and all other points in the *next nearest* cluster.
   - **Cluster Distribution:** The data should divide highly unevenly. Target counts are approximately:
     - Cluster 1: ~974 samples
     - Cluster 2: ~621 samples
     - Cluster 3: ~44 samples (outliers/tail)

---

## 4. Model 2: Self-Organizing Map (SOM) (Neural Clustering)
**Goal:** Map the 8-dimensional pulsar data onto a 2D topological grid to see if the clusters are disjoint or continuous.

### Implementation Steps:
1. **Build From Scratch:** Implement a Kohonen SOM using NumPy or TensorFlow. 
2. **Architecture:** 
   - Create a 2D grid of size **$5 \times 5$** (25 artificial neurons).
   - Set the number of training iterations (epochs) to **100**.
3. **Training Logic:**
   - For each pulsar, calculate the Euclidean distance to all 25 neurons.
   - Find the **Best-Matching Unit (BMU)**.
   - Update the weights of the BMU and its neighborhood, decaying the learning rate and radius exponentially over the 100 epochs.
4. **Target Benchmarks:**
   - **Cluster Distribution:** The map should converge such that only 3 grid cells (or 3 macro-neighborhoods) contain data, leaving the rest empty. Target counts are approximately:
     - Cell 1: ~1,079 samples
     - Cell 2: ~517 samples
     - Cell 3: ~43 samples
   - **Silhouette Score:** When calculating the score based on these 3 SOM clusters, it should also yield `~0.44`.

---

## 5. Model 3: Principal Component Analysis (PCA)
**Goal:** Reduce the 8D feature space to 2D for human visualization and manifold analysis. *Note: PCA is used ONLY for visualization, not prior to clustering.*

### Implementation Steps:
1. **Compute PCs:** Run PCA to extract the first two Principal Components (PC1 and PC2).
2. **Feature Correlation Analysis:**
   - Analyze which of the 8 features dictate the PCs.
   - *Expected Finding:* **DM-SNR Profile Skewness** and **DM-SNR Profile Mean** should overwhelmingly dominate the components.
3. **The "Boomerang" Plot:**
   - Create a 2D scatter plot mapping PC1 (X-axis) against PC2 (Y-axis).
   - *Expected Finding:* The scatter plot must reveal a distinct **"boomerang"** or V-shaped manifold.
4. **Visualizing Clusters:**
   - Generate one boomerang plot where the dots are color-coded by their K-Means cluster.
   - Generate a second boomerang plot where the dots are color-coded by their SOM cluster.

---

## 6. Scientific Interpretation & Defense Strategy
You must be prepared to defend *why* the clustering performance was relatively low (Silhouette = 0.44) and what the boomerang shape signifies.

**Key Talking Points for Q&A:**
1. **The Continuous Spectrum:** The boomerang manifold and the Silhouette score of 0.44 prove that the dataset does **not** contain perfectly separated, distinct clusters. The 1,639 pulsars form a continuous morphological spectrum.
2. **The "Anonymized" Data Problem:** In astronomy, pulsars are primarily categorized by their *spin frequency*. However, to create this dataset, scientists collapsed the radio data into an "integrated pulse profile". This completely stripped away the frequency data.
3. **Conclusion:** Because the crucial frequency feature is missing, the clustering algorithms were essentially analyzing shape variations, not astrophysical pulsar families. To achieve distinct clusters, raw, unprocessed frequency data would be required in future work.
