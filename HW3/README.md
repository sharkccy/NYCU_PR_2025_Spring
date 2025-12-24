# NYCU Pattern Recognition (Graduate) 2025 Spring HW3

- StudentID: 110612117

- Name: Chung-Yu Chang (張仲瑜)


# Introduction
This project conducts clustering analysis on the abalone dataset.

## Project Structure
- `main.py`: Main program containing data processing, experiment execution, and result visualization
- `my_clustering.py`: Custom implementations of clustering algorithms (K-means and Hierarchical)
- `requirements.txt`: Project dependencies
- `abalone/`: Dataset directory

## Experiment Content
1. Using 5 different clustering algorithms:
   - K-means
   - DBSCAN
   - Hierarchical Clustering
   - Spectral Clustering
   - Gaussian Mixture Model

2. Custom implementation of 2 algorithms:
   - K-means
   - Hierarchical Clustering (Average Linkage)

3. Evaluation Metrics:
   - Internal: Silhouette Coefficient
   - External: Adjusted Rand Index

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment:
```bash
python main.py
```

## Results Explanation
The program will output evaluation metrics for each algorithm and display 2D visualizations of clustering results.
- Silhouette Score: Range [-1, 1], closer to 1 indicates better clustering
- Adjusted Rand Index: Range [-1, 1], closer to 1 indicates better alignment with true labels 

See the `110612117_HW3.pdf` file for further discussion and analysis of the results.