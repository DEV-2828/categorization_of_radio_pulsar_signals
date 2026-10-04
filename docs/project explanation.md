# Project Explanation: Automated Identification and Sub-Categorization of Radio Pulsar Signal Candidates

## Problem Statement
Pulsars are highly magnetized, rapidly rotating neutron stars that emit beams of electromagnetic radiation from their magnetic poles. These beams sweep across space like a cosmic lighthouse, producing periodic pulse trains detectable by Earth-based radio telescopes. 

Modern wide-field surveys detect tens of millions of periodic candidate signals, where more than **90%** are caused by terrestrial Radio Frequency Interference (RFI) or background noise. Manual inspection is unscalable, making automated machine learning classification necessary.

The core machine learning challenges include:
1. **Severe Class Imbalance**: In the benchmark HTRU2 dataset, real pulsars constitute only ~9.16% of all recorded candidates (~1:10 imbalance ratio). A trivial majority-class classifier achieves ~90.8% accuracy with zero predictive utility.
2. **Asymmetric Error Costs (Recall vs. Precision)**: In observational astrophysics, the penalty for a False Negative (discarding a true pulsar discovery) is drastically higher than vetting a False Positive. Hence, **Recall / Sensitivity** and **PR-AUC** are prioritized over raw classification accuracy.
3. **Sub-Categorization via Unsupervised Exploration**: Beyond binary classification, true pulsar candidates exhibit internal morphology variations requiring unsupervised clustering and non-linear manifold analysis.

## Prerequisites
- **Language**: Python 3.x
- **Knowledge Requirements**:
  - Probability & Statistics (Generative and Discriminative models, Bayes' Theorem)
  - Supervised Machine Learning Algorithms (Gaussian Discriminant Analysis (GDA), Logistic Regression, Decision Trees, Random Forests)
  - Unsupervised Machine Learning Algorithms (K-Means Clustering, Principal Component Analysis (PCA), Kohonen Self-Organizing Maps)
  - Class imbalance handling strategies (Random Oversampling, SMOTE, class weights)
  - Model evaluation metrics (Precision, Recall, ROC-AUC, PR-AUC, Silhouette Scores)
- **Environment**: A working Python environment with `pip` or `conda` for dependency management.

## Requirements
- **Dataset**: HTRU2 Dataset (17,898 instances, 8 continuous numerical features). The CSV is headerless, meaning the columns must be parsed correctly without assuming the first row contains labels.
- **Python Libraries**: 
  - `pandas` for data manipulation
  - `numpy` for array and mathematical operations
  - `scikit-learn` for supervised/unsupervised models, preprocessing, and metrics
  - `matplotlib` and `seaborn` for plotting and visualization
  - `streamlit` for building the interactive web application dashboard
  - `plotly` for 2D/3D interactive visualizations

## Deliverables
1. **GitHub Repository**: A private GitHub Repo containing a clean modular directory layout, reproducible code, and a comprehensive `README.md` with installation instructions and an architecture diagram.
2. **Two-Page Write-Up**: A structured PDF document containing the Problem Statement, Dataset Specifications, Mathematical Methodology, Empirical Results, and Scientific Conclusions. Must include the HTRU2 dataset citation.
3. **Slide Deck**: A presentation (PPTX/PDF) with 8-10 slides summarizing the motivation, imbalance mitigation, model formulations (GDA vs LR vs RF), threshold tuning, unsupervised learning results, and a summary.
4. **Live Code Demonstration**: An interactive Streamlit web application that runs inference on pre-trained models, features a real-time threshold slider, confusion matrix dynamics, and 2D PCA cluster visualization.
5. **Review & Q&A Defense**: Oral defense justifying modeling choices, assumptions (e.g., GDA vs QDA), mitigation of data leakage during SMOTE, and domain-specific knowledge.

## Steps/Goals to Reach the Goal

### Phase 1: Environment Setup & Directory Architecture
- Establish a production-grade, modular Python project structure.
- Initialize directories for datasets (`htru2/`), source code (`src/`), saved models (`models/`), notebooks (`notebooks/`), web app (`app/`), and deliverables.

### Phase 2: Exploratory Data Analysis (EDA) & Preprocessing
- Load the headerless HTRU2 dataset and assign the 8 standard numerical column names and target label.
- Analyze distributions, skewness, kurtosis, and feature correlation (collinearity).
- Perform a Stratified 80/20 Train-Test Split (or Stratified K-Fold CV) to strictly prevent data leakage.
- Apply `StandardScaler` only on the training set and transform validation/test sets accordingly.
- Implement class imbalance strategies (ROS, SMOTE, `class_weight='balanced'`) specifically on the training fold, keeping the test set matching the original ~9.16% ratio.

### Phase 3: Supervised Classification Implementation
- **Baseline Models**: Build Gaussian Discriminant Analysis (GDA) from scratch and Logistic Regression using Scikit-Learn. Compare Generative vs. Discriminative approaches.
- **Primary Model**: Implement a tuned Random Forest Classifier via Scikit-Learn (`RandomizedSearchCV`) and a secondary from-scratch binary Decision Tree/Random Forest using Information Gain.
- **Threshold Calibration**: Shift the decision threshold (e.g., down to $\tau \approx 0.20$) to heavily optimize Recall.
- **Evaluation**: Evaluate and compare models on the unseen test set using Precision, Recall, PR-AUC, and ROC-AUC. Produce Feature Importance plots.

### Phase 4: Unsupervised Exploration & Manifold Analysis
- Filter the dataset to include only the 1,639 positive true pulsar samples.
- **Baseline Clustering**: Execute K-Means clustering. Validate the optimal cluster count using the Elbow Method and Silhouette Scores (target $k=3$).
- **Neural Clustering**: Implement a 2D Kohonen Self-Organizing Map (SOM, e.g., 5x5 grid) from scratch and evaluate its topology-preserving clusters visually (U-Matrix) and quantitatively.
- **Dimensionality Reduction**: Use Principal Component Analysis (PCA) to project the 8D cluster space into 2 Principal Components, uncovering the "boomerang" manifold shape that describes continuous morphology variations.

### Phase 5: Interactive Web Application (Streamlit)
- Develop an interactive demonstration application (`app/app.py`) relying purely on pre-trained serialized models.
- Implement a live prediction engine with interactive sliders for the 8 features.
- Add an interactive threshold slider to instantly recalculate metrics, Precision, Recall, and the confusion matrix.
- Integrate an interactive 2D/3D PCA manifold scatter viewer with hoverable cluster data, plus a model comparison tab.

### Phase 6: Final Documentation & Defense Preparation
- Serialize trained models (e.g., `.pkl` files) into the `models/` directory.
- Generate and save required plots (`figures/`).
- Complete the 2-page executive summary write-up and the presentation slide deck.
- Prepare for the oral defense by mastering the theoretical assumptions behind each model, how data leakage was handled, and the astrophysical context of the performance metrics.
