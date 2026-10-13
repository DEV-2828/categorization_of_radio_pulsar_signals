# Unsupervised Pulsar Categorization — Presentation Guide & Dashboard

This directory contains the presentation dashboard for the **Unsupervised Categorization of Radio Pulsar Signals** (SEM-5 Machine Learning project).

---

## 🚀 How to Launch the Presentation

### Method 1: One-Click Batch File (Windows)
Double-click [`run_presentation.bat`](file:///c:/DEVOPAM/ENGINEERING/SEM-5/SEM_5_CODES/ML/ML_JACKFRUIT/UNSUPERVISED_categorization_of_pulsars/PRESENTATION/run_presentation.bat).

### Method 2: Command Line
Open a terminal in this directory and run:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧭 Dashboard Sections & Presentation Flow

| # | Section | Key Talking Points for Evaluator |
|---|---|---|
| **1** | **Scope & Problem Definition** | • Explain that your team members handled *Pulsar vs. Noise* classification, while **your role is discovering the hidden sub-types of the 1,639 confirmed true pulsars**.<br>• Ground-truth labels are lost during folding and phase averaging, making unsupervised learning scientifically necessary.<br>• Justify training on **unscaled features** to preserve physical variance and match Debesai et al. |
| **2** | **PCA & The Boomerang Manifold** | • Show that **2 principal components explain 86.04% of total variance** (PC1: 61.06%, PC2: 24.98%).<br>• Show the interactive 2D/3D scatter plot of the **"Boomerang" Manifold**.<br>• Explain the physics: Left arm is driven by decreasing `mean_dmsnr_curve`, right arm is driven by increasing `skewness_dmsnr_curve`.<br>• The tip of the boomerang belongs exclusively to Cluster 3. |
| **3** | **K-Means Sub-populations ($k=3$)** | • Show the **Elbow Curve** and **Silhouette Curve** peaking at $k=3$.<br>• Silhouette Score: **0.4413** (paper target: ~0.44 — deviation only +0.3%).<br>• Walk through the 3 discovered populations:<br>  - **Cluster 1 ($n=974$):** Canonical "Normal" pulsars (medium brightness, moderate DM-SNR).<br>  - **Cluster 2 ($n=621$):** Distant / heavily dispersed pulsars (dimmer pulses, huge DM-SNR responses).<br>  - **Cluster 3 ($n=44$):** Nearby / millisecond candidates (bright, razor-sharp DM-SNR peak, extreme skewness $255.6$).<br>• Show the **8D Radar Chart** to compare cluster profiles visually. |
| **4** | **Self-Organizing Map (SOM)** | • Implemented from scratch using **NumPy** ($5 \times 5$ grid, 100 epochs, decaying learning rate).<br>• Discuss the **Macro-cluster Collapse phenomenon**: why the rare 3rd cluster ($2.7\%$ of data) was absorbed during the neuron-to-centroid assignment step, leading to Silhouette of 0.4024. |
| **5** | **Live Pulsar Sandbox** | • **Interactive Demonstration:** Select a real pulsar by index or use sliders to adjust parameters live.<br>• Watch the candidate dynamically project onto the Boomerang manifold with instant cluster classification. |
| **6** | **Paper Parity Scorecard** | • Side-by-side comparison with **Debesai, Gutierrez, & Koyluoglu (Stanford CS 229)**.<br>• Highlights exact parity in cluster sizes (974, 621, 44), Silhouette score (0.4413 vs 0.44), and top PCA driver features. |

---

## 📦 Files

- [`app.py`](file:///c:/DEVOPAM/ENGINEERING/SEM-5/SEM_5_CODES/ML/ML_JACKFRUIT/UNSUPERVISED_categorization_of_pulsars/PRESENTATION/app.py): Complete interactive Streamlit dashboard.
- [`data_loader.py`](file:///c:/DEVOPAM/ENGINEERING/SEM-5/SEM_5_CODES/ML/ML_JACKFRUIT/UNSUPERVISED_categorization_of_pulsars/PRESENTATION/data_loader.py): Cached data extraction, PCA transformation, and cluster mappings.
- [`run_presentation.bat`](file:///c:/DEVOPAM/ENGINEERING/SEM-5/SEM_5_CODES/ML/ML_JACKFRUIT/UNSUPERVISED_categorization_of_pulsars/PRESENTATION/run_presentation.bat): Windows batch launch script.
