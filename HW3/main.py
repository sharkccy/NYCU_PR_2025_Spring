import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from my_clustering import MyKMeans, MyAgglomerativeClustering, MyDBSCAN

import os
import time

def load_and_preprocess_data(file_path, use_sex_as_label=False):
    """Load and preprocess the data"""
    # Read data
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    
    # Convert sex to numerical values
    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    data['Sex'] = data['Sex'].map(sex_mapping)
    
    # Standardize features
    scaler = StandardScaler()
    if use_sex_as_label:
        X = scaler.fit_transform(data.iloc[:, 1:9])  # 排除性別（因為是標籤）
        true_labels = data['Sex']
    else:
        X = scaler.fit_transform(data.iloc[:, 0:8])  # 排除年齡環（因為是標籤）
        # 使用等頻分bin（確保每個bin中的樣本數量相近）
        n_bins = 5  # 保持5個分類
        true_labels = pd.qcut(data['Rings'], q=n_bins, labels=False)
        
        # 計算並輸出每個bin的樣本數量
        bin_counts = pd.value_counts(true_labels, sort=False)
        bin_edges = pd.qcut(data['Rings'], q=n_bins).cat.categories
        print("\n各年齡層的樣本分布（等頻分bin）：")
        for bin_idx, count in bin_counts.items():
            bin_range = bin_edges[bin_idx]
            print(f"Bin {bin_idx} (Rings {bin_range}): {count} 個樣本")
    
    return X, true_labels, data

def evaluate_clustering(X, labels, true_labels):
    """Evaluate clustering results"""
    silhouette = silhouette_score(X, labels)
    ari = adjusted_rand_score(true_labels, labels)
    return silhouette, ari

# def plot_clusters_2d(X, labels, title):
#     """Plot clustering results in 2D using PCA"""
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
#     plt.title(title)
#     plt.xlabel('First Principal Component')
#     plt.ylabel('Second Principal Component')
#     plt.colorbar(label='Cluster')
#     plt.show()

def plot_scores(n_values, silhouette_scores, ari_scores, x_label, title):
    """Plot silhouette and ARI scores"""
    plt.figure(figsize=(10, 5))
    
    # 創建雙Y軸
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    # 繪製Silhouette Score，使用點標記和虛線
    line1 = ax1.plot(n_values, silhouette_scores, 'b', label='Silhouette Score', 
                     linestyle='--', marker='o', markersize=8)
    ax1.set_ylabel('Silhouette Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 繪製ARI，使用點標記和虛線
    line2 = ax2.plot(n_values, ari_scores, 'r', label='ARI', 
                     linestyle='--', marker='o', markersize=8)
    ax2.set_ylabel('Adjusted Rand Index', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 設置x軸
    ax1.set_xlabel(x_label)
    ax1.set_xticks(n_values)
    
    # 添加圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_categorical_scores(categories, silhouette_scores, ari_scores, x_label, title):
    """Plot silhouette and ARI scores for categorical parameters using bar plots"""
    plt.figure(figsize=(10, 5))
    
    # 設置柱狀圖的位置
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    # 繪製Silhouette Score
    rects1 = ax1.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', color='b', alpha=0.7)
    ax1.set_ylabel('Silhouette Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 繪製ARI
    rects2 = ax2.bar(x + width/2, ari_scores, width, label='ARI', color='r', alpha=0.7)
    ax2.set_ylabel('Adjusted Rand Index', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 設置x軸
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_xlabel(x_label)
    
    # 添加數值標籤
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
    
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    
    # 添加圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}.png')
    plt.close()

def experiment_kmeans(X, true_labels):
    """Experiment with KMeans parameters"""
    print("\nKMeans Experiments:")
    print("-" * 50)
    
    # Experiment 1: Different number of clusters
    n_clusters_range = range(2, 21)
    silhouette_scores = []
    ari_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"n_clusters={n_clusters}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_scores(list(n_clusters_range), silhouette_scores, ari_scores, 
               'Number of Clusters', 'KMeans - Number of Clusters vs Scores')
    
    # Experiment 2: Different initialization methods
    init_methods = ['k-means++', 'random']
    n_clusters = 3  # 使用一個固定的群集數
    silhouette_scores = []
    ari_scores = []
    
    for init in init_methods:
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=int(time.time()))
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"initialization={init}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_categorical_scores(init_methods, silhouette_scores, ari_scores,
                          'Initialization Method', 'KMeans - Initialization Methods vs Scores')

def experiment_dbscan(X, true_labels):
    """Experiment with DBSCAN parameters"""
    print("\nDBSCAN Experiments:")
    print("-" * 50)
    
    # Experiment 1: Different eps values
    eps_range = np.linspace(0.1, 2.0, 20)
    silhouette_scores = []
    ari_scores = []
    valid_eps = []
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=32)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:  # 確保至少有兩個群集
            silhouette = silhouette_score(X, labels)
            ari = adjusted_rand_score(true_labels, labels)
            silhouette_scores.append(silhouette)
            ari_scores.append(ari)
            valid_eps.append(eps)
            print(f"eps={eps:.2f}:")
            print(f"Number of clusters: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print("-" * 30)
    
    if valid_eps:
        plot_scores(valid_eps, silhouette_scores, ari_scores, 
                   'Epsilon', 'DBSCAN - Epsilon vs Scores')
    
    # Experiment 2: Different min_samples
    min_samples_range = range(2, 100, 10)
    silhouette_scores = []
    ari_scores = []
    valid_min_samples = []
    
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            ari = adjusted_rand_score(true_labels, labels)
            silhouette_scores.append(silhouette)
            ari_scores.append(ari)
            valid_min_samples.append(min_samples)
            print(f"min_samples={min_samples}:")
            print(f"Number of clusters: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print("-" * 30)
    
    if valid_min_samples:
        plot_scores(valid_min_samples, silhouette_scores, ari_scores, 
                   'Min Samples', 'DBSCAN - Min Samples vs Scores')

def experiment_hierarchical(X, true_labels):
    """Experiment with Hierarchical Clustering parameters"""
    print("\nHierarchical Clustering Experiments:")
    print("-" * 50)
    
    # Experiment 1: Different linkage criteria
    linkages = ['ward', 'complete', 'average']
    n_clusters = 8  # 固定群集數
    silhouette_scores = []
    ari_scores = []
    
    for linkage in linkages:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"linkage={linkage}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_categorical_scores(linkages, silhouette_scores, ari_scores,
                            'Linkage Criterion', 'Hierarchical - Linkage Methods vs Scores')
    
    # Experiment 2: Different number of clusters
    n_clusters_list = range(2, 21, 2)
    silhouette_scores = []
    ari_scores = []
    
    for n_clusters in n_clusters_list:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        labels = hierarchical.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"n_clusters={n_clusters}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_scores(n_clusters_list, silhouette_scores, ari_scores,
               'Number of Clusters', 'Hierarchical - Number of Clusters vs Scores')



def experiment_meanshift(X, true_labels):
    """Experiment with Mean Shift Clustering parameters"""
    print("\nMean Shift Experiments:")
    print("-" * 50)
    
    # Experiment with different bandwidth values
    # 先用estimate_bandwidth得到一個基準值
    bandwidth_base = estimate_bandwidth(X, quantile=0.2)
    print(f"Bandwidth base: {bandwidth_base}")  
    
    # 測試不同的bandwidth倍數
    bandwidth_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    silhouette_scores = []
    ari_scores = []
    
    for mult in bandwidth_multipliers:
        bandwidth = bandwidth_base * mult
        meanshift = MeanShift(bandwidth=bandwidth)
        labels = meanshift.fit_predict(X)
        
        # 確保至少有兩個群集
        n_clusters = len(set(labels))
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            ari = adjusted_rand_score(true_labels, labels)
            silhouette_scores.append(silhouette)
            ari_scores.append(ari)
            print(f"bandwidth_multiplier={mult:.2f}:")
            print(f"Number of clusters: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print("-" * 30)
    
    plot_scores(bandwidth_multipliers, silhouette_scores, ari_scores,
               'Bandwidth Multiplier', 'MeanShift - Bandwidth vs Scores')
    
def experiment_gmm(X, true_labels):
    """Experiment with Gaussian Mixture Model parameters"""
    print("\nGaussian Mixture Model Experiments:")
    print("-" * 50)
    
    # Experiment 1: Different number of components
    n_components_list = np.arange(2, 21, 2)
    silhouette_scores = []
    ari_scores = []
    
    for n_components in n_components_list:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"n_components={n_components}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_scores(n_components_list, silhouette_scores, ari_scores,
               'Number of Components', 'GMM - Number of Components vs Scores')
    
    # Experiment 2: Different covariance types
    covariance_types = ['full', 'spherical']
    n_components = 4 # 固定組件數
    silhouette_scores = []
    ari_scores = []
    
    for covariance_type in covariance_types:
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        labels = gmm.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"covariance_type={covariance_type}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_categorical_scores(covariance_types, silhouette_scores, ari_scores,
                            'Covariance Type', 'GMM - Covariance Types vs Scores')

def experiment_my_kmeans(X, true_labels):
    """Experiment with custom KMeans implementation"""
    print("\nMy KMeans Experiments:")
    print("-" * 50)
    
    # Experiment with different number of clusters
    n_clusters_range = range(2, 21, 2)
    silhouette_scores = []
    ari_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = MyKMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"n_clusters={n_clusters}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_scores(list(n_clusters_range), silhouette_scores, ari_scores, 
               'Number of Clusters', 'MyKMeans - Number of Clusters vs Scores')

def experiment_my_hierarchical(X, true_labels):
    """Experiment with custom Hierarchical Clustering implementation"""
    print("\nMy Hierarchical Clustering Experiments:")
    print("-" * 50)
    
    # Experiment with different number of clusters
    n_clusters_list = range(2, 21, 2)
    silhouette_scores = []
    ari_scores = []
    
    for n_clusters in n_clusters_list:
        hierarchical = MyAgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        ari = adjusted_rand_score(true_labels, labels)
        silhouette_scores.append(silhouette)
        ari_scores.append(ari)
        print(f"n_clusters={n_clusters}:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print("-" * 30)
    
    plot_scores(n_clusters_list, silhouette_scores, ari_scores,
               'Number of Clusters', 'MyHierarchical - Number of Clusters vs Scores')

def experiment_my_dbscan(X, true_labels):
    """Experiment with custom DBSCAN implementation"""
    print("\nMy DBSCAN Experiments:")
    print("-" * 50)
    
    # Experiment with different eps values
    eps_range = np.linspace(0.1, 2.0, 20)
    min_samples = 32  # 固定min_samples
    silhouette_scores = []
    ari_scores = []
    valid_eps = []
    
    for eps in eps_range:
        dbscan = MyDBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:  # 確保至少有兩個群集
            silhouette = silhouette_score(X, labels)
            ari = adjusted_rand_score(true_labels, labels)
            silhouette_scores.append(silhouette)
            ari_scores.append(ari)
            valid_eps.append(eps)
            print(f"eps={eps:.2f}:")
            print(f"Number of clusters: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print("-" * 30)
    
    if valid_eps:
        plot_scores(valid_eps, silhouette_scores, ari_scores, 
                   'Epsilon', 'MyDBSCAN - Epsilon vs Scores')

def main():
    # Load and preprocess data
    X, true_labels, data = load_and_preprocess_data('abalone/abalone.data', use_sex_as_label=False)  # 設定要使用哪種標籤
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Run experiments
    # experiment_kmeans(X, true_labels)
    # experiment_dbscan(X, true_labels)
    # experiment_hierarchical(X, true_labels)
    # experiment_meanshift(X, true_labels)
    # experiment_gmm(X, true_labels)
    # experiment_my_kmeans(X, true_labels)
    # experiment_my_hierarchical(X, true_labels)
    experiment_my_dbscan(X, true_labels)

if __name__ == '__main__':
    main() 