# PCA-KMeans-Classification

# Project Overview

This project implements custom Principal Component Analysis (PCA) and K-Means clustering algorithms to classify "Cancer" and "Normal" samples using the ABIDE II dataset. It also includes validation on the Iris dataset to compare custom PCA with scikit-learn's PCA.

# Usage
1. Clone the repository:
- git clone https://github.com/nvhungus/PCA-KMeans-Classification.git

2. Install required dependencies:
- pip install pandas numpy matplotlib seaborn scikit-learn

3. Place the ABIDE2.csv dataset in the project directory.
4. Run the Jupyter notebook:
- jupyter notebook ABIDE-II-Clustering.ipynb

5. Execute the notebook cells to preprocess data, apply PCA and K-Means, and visualize results.

# Algorithm
- PCA: Custom implementation to reduce dimensionality by projecting data onto principal components. Features include data centering, SVD decomposition, and explained variance ratio calculation.
- K-Means: Custom implementation with Euclidean distance and k-means++ initialization. Supports multiple runs (n_init=100) to select the best clustering based on inertia.
- Preprocessing: Handles missing values (filled with mean), removes outliers using IQR-based feature fraction, and standardizes data (Z-score).
- Evaluation: Computes accuracy, precision, recall, F1-score, Adjusted Rand Index (ARI), Silhouette score, and confusion matrix.

# Model Performance
The final model with 20 PCA components achieves:
- Accuracy: 55.44%
- Precision: 51.13%
- Recall: 73.75%
- F1-score: 60.39%
- ARI: 1.03%
- Silhouette: 4.19%
- Confusion Matrix: TP = 340, TN = 215, FP = 325, FN = 121
- Inertia: 19300.59

# Summary

The project demonstrates a full pipeline for unsupervised classification using PCA and K-Means. It preprocesses the ABIDE II dataset, reduces dimensionality, clusters data, and evaluates performance. The custom PCA closely matches scikit-learn's PCA on the Iris dataset, validating its correctness.

# Conclusion

The model achieves moderate accuracy (55.44%) in classifying "Cancer" and "Normal" samples, with a high recall (73.75%) but low ARI and Silhouette scores, indicating limited cluster separation. Future improvements could include feature selection, alternative clustering methods, or supervised learning approaches.
