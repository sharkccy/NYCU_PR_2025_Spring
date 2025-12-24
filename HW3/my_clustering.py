import numpy as np

class MyKMeans:
    def __init__(self, n_clusters=10, max_iters=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.labels_ = None
        self.centroids = None
        
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for _ in range(self.max_iters):
            # Assign samples to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < 1e-6):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        # Assign labels to new data
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

class MyAgglomerativeClustering:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.labels_ = None
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # Initialize distance matrix
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.sqrt(((X[i] - X[j])**2).sum())
                distances[j, i] = distances[i, j]
        
        # Initialize: each sample as a cluster
        self.labels_ = np.arange(n_samples)
        current_clusters = [{i} for i in range(n_samples)]
        
        while len(current_clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i = merge_j = 0
            
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    # Calculate average linkage distance between clusters
                    cluster_dist = 0
                    for idx1 in current_clusters[i]:
                        for idx2 in current_clusters[j]:
                            cluster_dist += distances[idx1, idx2]
                    cluster_dist /= (len(current_clusters[i]) * len(current_clusters[j]))
                    
                    if cluster_dist < min_dist:
                        min_dist = cluster_dist
                        merge_i, merge_j = i, j
            
            # Merge closest clusters
            current_clusters[merge_i].update(current_clusters[merge_j])
            current_clusters.pop(merge_j)
            
            # Update labels
            new_label = min(self.labels_[list(current_clusters[merge_i])])
            for idx in current_clusters[merge_i]:
                self.labels_[idx] = new_label
                
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_ 

class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def _get_neighbors(self, X, point_idx):
        # 計算點到所有其他點的距離，返回在eps範圍內的點的索引
        distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))
        return np.where(distances <= self.eps)[0]
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # 初始化所有點為噪聲點
        cluster_label = 0
        
        # 對每個點進行處理
        for point_idx in range(n_samples):
            if self.labels_[point_idx] != -1:  # 跳過已經被訪問的點
                continue
                
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:  # 不是核心點
                continue
                
            # 發現新的群集
            self.labels_[point_idx] = cluster_label
            
            # 擴展群集
            seed_points = neighbors.tolist()
            while seed_points:
                current_point = seed_points.pop()
                if self.labels_[current_point] == -1:  # 未訪問的點
                    self.labels_[current_point] = cluster_label
                    current_neighbors = self._get_neighbors(X, current_point)
                    if len(current_neighbors) >= self.min_samples:
                        seed_points.extend([p for p in current_neighbors if self.labels_[p] == -1])
            
            cluster_label += 1
            
        return self.labels_