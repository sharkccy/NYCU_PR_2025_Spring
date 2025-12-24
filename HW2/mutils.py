import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import KernelPCA

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("Columns:", df.columns.tolist())
    if file_path == "data/obesity.csv":
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
        df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
        df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
        df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
        df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        df['MTRANS'] = df['MTRANS'].map({'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Automobile': 4})
        df['NObeyesdad'] = df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6})
    
        X = df.drop(columns=['NObeyesdad'], axis=1).values
        Y = df['NObeyesdad'].values
        num_classes = len(np.unique(Y))

        print(X)
        print('---')
        print(Y)
        print('---')
        print(num_classes)
        print("---")

    elif file_path == "data/churn.csv":
        X = df.drop(columns=['Churn'], axis=1).values
        Y = df['Churn'].values
        num_classes = len(np.unique(Y))

        print(X)
        print('---')
        print(Y)
        print('---')
        print(num_classes)
        print("---")

    elif file_path == "data/forest.csv":
        df['Classes'] = df['Classes'].map({'not fire': 0, 'fire': 1})
        X = df.drop(columns=['Classes'], axis=1).values
        Y = df['Classes'].values
        num_classes = len(np.unique(Y))

        print(X)
        print('---')
        print(Y)
        print('---')
        print(num_classes)
        print("---")
            
    elif file_path == "data/wine.csv":
        df = pd.read_csv(file_path, sep=';')
        df.columns = df.columns.str.strip()
        print("Columns:", df.columns.tolist()) 

        df['quality'] = df['quality'].map({3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5})
        X = df.drop(columns=['quality'], axis=1).values
        Y = df['quality'].values
        num_classes = len(np.unique(Y))

        print(X)
        print('---')
        print(Y)
        print('---')
        print(num_classes)
        print("---")

    stds = np.std(X, axis=0)
    if np.any(stds == 0):
        # print("Warning: Some features have zero variance, replacing std with 1 to avoid division by zero")
        zero_variance_indices = np.where(stds == 0)[0]
        X = np.delete(X, zero_variance_indices, axis=1)
        stds = np.std(X, axis=0)
        print(f"Removed features with zero variance: {zero_variance_indices}")

    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 

    print("data has been loaded and preprocessed!")
    return X, Y, num_classes

def k_fold_cross_validation(n_samples, k=5, shuffle=True, random_seed = 42):
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop

    return folds

def split_data(X, Y, k=5, shuffle=True, random_seed=42, stratify=True):
    np.random.seed(random_seed)
    n_samples = len(Y)
    indices = np.arange(n_samples)
    if stratify:
        # 分層抽樣
        classes, counts = np.unique(Y, return_counts=True)
        class_indices = {cls: np.where(Y == cls)[0] for cls in classes}
        test_size = n_samples // k
        test_indices = []
        train_val_indices = []

        # 計算每個類別的測試集樣本數
        total_counts = sum(counts)
        test_proportions = {cls: count / total_counts for cls, count in zip(classes, counts)}
        test_samples_per_class = {cls: int(round(test_size * prop)) for cls, prop in test_proportions.items()}

        # 確保至少分配一個樣本給測試集，並調整總數
        for cls in classes:
            if test_samples_per_class[cls] == 0 and counts[classes == cls][0] > 0:
                test_samples_per_class[cls] = 1
        current_test_size = sum(test_samples_per_class.values())
        if current_test_size < test_size:
            # 若總數不足，隨機增加樣本到某類別
            remaining = test_size - current_test_size
            for _ in range(remaining):
                cls = np.random.choice(classes)
                if test_samples_per_class[cls] < counts[classes == cls][0]:
                    test_samples_per_class[cls] += 1

        # 為每個類別分配測試集和訓練/驗證集索引
        for cls in classes:
            cls_indices = class_indices[cls]
            n_test = test_samples_per_class[cls]
            if shuffle:
                np.random.shuffle(cls_indices)
            test_indices.extend(cls_indices[:n_test])
            train_val_indices.extend(cls_indices[n_test:])

        test_indices = np.array(test_indices)
        train_val_indices = np.array(train_val_indices)

        if shuffle:
            np.random.shuffle(test_indices)
            np.random.shuffle(train_val_indices)

    else:
        if shuffle:
            np.random.shuffle(indices)

        test_size = n_samples // k
        test_indices = indices[:test_size]
        train_val_indices = indices[test_size:]

    X_train_val, Y_train_val = X[train_val_indices], Y[train_val_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    train_val_split = k_fold_cross_validation(len(train_val_indices), k=k, shuffle=True, random_seed=random_seed)
    
    X_train_val = (X_train_val - np.mean(X_train_val, axis=0)) / np.std(X_train_val, axis=0)
    X_test = (X_test - np.mean(X_train_val, axis=0)) / np.std(X_train_val, axis=0)
    
    print("Class distribution in full dataset:", np.bincount(Y))
    print("Class distribution in test set:", np.bincount(Y_test))
    print("Class distribution in train/val set:", np.bincount(Y[train_val_indices]))
    
    return X_test, Y_test, train_val_indices, train_val_split, X_train_val, Y_train_val

def compute_kernel_pca_with_variance_selection(X, variance_threshold=[0.25, 0.50, 0.75, 0.90], kernel='rbf', gamma=0.1):
    max_feature = min(X.shape[0], X.shape[1])

    kpca = KernelPCA(kernel=kernel, gamma=gamma, n_components=max_feature)
    X_kpca = kpca.fit_transform(X)
    eigenvalues = kpca.eigenvalues_
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    pca_dims = []
    for var_pct in variance_threshold:
        n_components = int(np.argmax(cumulative_variance >= var_pct) + 1)
        if n_components == 0:
            n_components = 1
        pca_dims.append(n_components)

    pca_dims.append(max_feature)
    print(f"Number of components for 100% variance: {max_feature}")
    pca_dims = sorted(list(set(pca_dims)))

    pca_results = {}
    for n_components in pca_dims:
        kpca = KernelPCA(kernel=kernel, gamma=gamma, n_components=n_components)
        X_pca = kpca.fit_transform(X)
        var_ratio = explained_variance_ratio[:n_components]
        pca_results[n_components] = (X_pca, kpca, var_ratio)

    return pca_results, pca_dims, eigenvalues

def compute_linear_pca_with_variance_selection(X, variance_threshold = [0.25, 0.5, 0.75, 0.9], num_components=None):
    # print("X_mean:", np.mean(X, axis=0))
    # print("X_std:", np.std(X, axis=0))

    covariance_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    if num_components is not None:
        pca_results = {}
        W = sorted_eigenvectors[:, :num_components]
        X_pca = np.dot(X, W)
        var_ratio = explained_variance_ratio[:num_components]
        pca_results = (X_pca, W, var_ratio)
        return pca_results, num_components, sorted_eigenvalues
    else:
        pca_dims = []
        for threshold in variance_threshold:
            n_components = int(np.argmax(cumulative_variance >= threshold) + 1)
            if n_components == 0:
                n_components = 1 
            pca_dims.append(n_components)
            print(f"Number of components for {threshold*100:.1f}% variance: {n_components}")

        max_feature = X.shape[1]
        pca_dims.append(max_feature)
        print(f"Number of components for 100% variance: {max_feature}")
        pca_dims = sorted(list(set(pca_dims)))

        pca_results = {}
        for n_components in pca_dims:
            W = sorted_eigenvectors[:, :n_components]
            X_pca = np.dot(X, W)
            var_ratio = explained_variance_ratio[:n_components]
            pca_results[n_components] = (X_pca, W, var_ratio)

        return pca_results, pca_dims, sorted_eigenvalues

def compute_lda(X, Y, num_components=None):
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction.
    X: [n_samples, n_features] — 已標準化資料
    Y: [n_samples] — 類別標籤
    num_components: 想要保留的維度數，預設為 C - 1
    """
    n_samples, n_features = X.shape
    class_labels = np.unique(Y)
    n_classes = len(class_labels)

    if num_components is None:
        num_components = n_classes - 1

    mean_overall = np.mean(X, axis=0)

    S_w = np.zeros((n_features, n_features))
    S_b = np.zeros((n_features, n_features))

    for c in class_labels:
        X_c = X[Y == c]
        mean_c = np.mean(X_c, axis=0)
        S_w += (X_c - mean_c).T @ (X_c - mean_c)
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
        S_b += n_c * (mean_diff @ mean_diff.T)

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)

    sorted_indices = np.argsort(np.real(eigvals))[::-1]
    topk_eigvecs = np.real(eigvecs[:, sorted_indices[:num_components]])

    X_lda = X @ topk_eigvecs

    return X_lda, topk_eigvecs

# def compute_pca(X, n_components):
#     X_centered = X - np.mean(X, axis=0)
#     covariance_matrix = np.cov(X_centered, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

#     sorted_indices = np.argsort(eigenvalues)[::-1]
#     sorted_eigenvalues = eigenvalues[sorted_indices]
#     sorted_eigenvectors = eigenvectors[:, sorted_indices]

#     W = sorted_eigenvectors[:, :n_components]
#     X_pca = np.dot(X_centered, W)

#     # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
#     # sorted_eigenvalues = S**2 / (X.shape[0] - 1)
#     # W = Vt.T[:, :n_components]
#     # X_pca = U * S

#     return X_pca, W, sorted_eigenvalues

def compute_mutual_information(X, Y):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(Y))
    mi = np.zeros(n_features)

    for f in range(n_features):
        X_feature = X[:, f]
        bins = np.linspace(np.min(X_feature), np.max(X_feature), 16)
        X_feature_binned = np.digitize(X_feature, bins) - 1
        X_feature_binned = np.clip(X_feature_binned, 0, len(bins) - 2)  

        joint_hist = np.zeros((n_classes, len(bins) - 1))  
        for i in range(n_samples):
            joint_hist[Y[i], X_feature_binned[i]] += 1
        joint_hist /= n_samples

        p_y = np.sum(joint_hist, axis=1)  # Y 的邊緣分佈
        p_x = np.sum(joint_hist, axis=0)  # X 的邊緣分佈

        mi_feature = 0
        for y in range(n_classes):
            for x in range(len(bins) - 1):
                if joint_hist[y, x] > 0 and p_y[y] > 0 and p_x[x] > 0:
                    mi_feature += joint_hist[y, x] * np.log(joint_hist[y, x] / (p_y[y] * p_x[x]))

        mi[f] = mi_feature

    return mi
    
def select_features(X, Y, n_features):
    mi = compute_mutual_information(X, Y)
    top_indices = np.argsort(mi)[::-1][:n_features]
    X_selected = X[:, top_indices]
    return X_selected, top_indices

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()


def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = np.trace(cm) / np.sum(cm)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    total_f1_score = np.mean(f1_score)


    return accuracy, total_precision, total_recall, total_f1_score


def plot_metrics(history, title, filename):
    plt.figure(figsize=(12, 10))
    folds = range(1, len(history['train_acc']) + 1)

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.scatter(folds, history['train_acc'], label='Train Accuracy', marker='o')
    plt.scatter(folds, history['val_acc'], label='Validation Accuracy', marker='x')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.xticks(folds)
    plt.legend()

    # Loss
    plt.subplot(2, 2, 2)
    plt.scatter(folds, history['train_loss'], label='Train Loss', marker='o')
    plt.scatter(folds, history['val_loss'], label='Validation Loss', marker='x')
    plt.title(f'{title} - Loss')
    plt.xlabel('Folds')
    plt.ylabel('Loss')
    plt.xticks(folds)
    plt.legend()

    # F1 Score
    plt.subplot(2, 2, 3)
    plt.scatter(folds, history['train_f1'], label='Train F1 Score', marker='o')
    plt.scatter(folds, history['val_f1'], label='Validation F1 Score', marker='x')
    plt.title(f'{title} - F1 Score')
    plt.xlabel('Folds')
    plt.ylabel('F1 Score')
    plt.xticks(folds)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_roc_auc(y_true, y_score, num_classes):
    y_true_one_hot = np.eye(num_classes)[y_true]
    fpr, tpr, auc = [], [], []

    for i in range(num_classes):
        sorted_indices = np.argsort(y_score[:, i])[::-1] # [:, i] for all rows, i-th column, [::-1] for descending order, start, end, step
        y_true_sorted = y_true_one_hot[:, i][sorted_indices] # rearranging the order along the column in y_true_one_hot according to the sorted indices
        y_score_sorted = y_score[:, i][sorted_indices]

        tp, fp = 0, 0
        tpr_i, fpr_i = [0], [0]
        total_positives = np.sum(y_true_one_hot[:, i])
        total_negatives = len(y_true_sorted) - total_positives

        for j in range(len(y_true_sorted)):
            if y_true_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            tpr_i.append(tp / total_positives if total_positives > 0 else 0)
            fpr_i.append(fp / total_negatives if total_negatives > 0 else 0)

        fpr.append(fpr_i)
        tpr.append(tpr_i)
        # 使用梯形法則計算 AUC
        auc_i = np.trapz(tpr_i, fpr_i)
        auc.append(auc_i if auc_i >= 0 else 0)  # 確保不小於 0

    # Micro ROC 和 Micro AUC
    y_true_flat = y_true_one_hot.ravel()
    y_score_flat = y_score.ravel()
    sorted_indices = np.argsort(y_score_flat)[::-1]
    y_true_sorted = y_true_flat[sorted_indices]
    y_score_sorted = y_score_flat[sorted_indices]

    tp, fp = 0, 0
    tpr_micro, fpr_micro = [0], [0]
    total_positives = np.sum(y_true_flat)
    total_negatives = len(y_true_flat) - total_positives

    for j in range(len(y_true_sorted)):
        if y_true_sorted[j] == 1:
            tp += 1
        else:
            fp += 1
        tpr_micro.append(tp / total_positives if total_positives > 0 else 0)
        fpr_micro.append(fp / total_negatives if total_negatives > 0 else 0)

    auc_micro = np.trapz(tpr_micro, fpr_micro)
    auc_micro = auc_micro if auc_micro >= 0 else 0  # 確保不小於 0

    return fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro

def plot_roc_curve(fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro, num_classes, title, filename):
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {auc[i]:.2f})')
    plt.plot(fpr_micro, tpr_micro, label=f'Micro (Weighted ROC, AUC = {auc_micro:.2f})', linewidth=2, linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_fs_comparison(feature_nums, gnb_fs_results, mlr_fs_results):
    plt.figure(figsize=(12, 8))
    handles = []
    labels = []

    for n in feature_nums:
        line, = plt.plot(gnb_fs_results[n]['fpr_micro'], gnb_fs_results[n]['tpr_micro'], 
                         label=f"GNB FS {n} (Prec = {gnb_fs_results[n]['prec']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB FS {n} (Prec = {gnb_fs_results[n]['prec']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
        line, = plt.plot(mlr_fs_results[n]['fpr_micro'], mlr_fs_results[n]['tpr_micro'], 
                         label=f"MLR FS {n} (Acc = {mlr_fs_results[n]['prec']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR FS {n} (Prec = {mlr_fs_results[n]['prec']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")

    def extract_sort_key(label):
        parts = label.split()
        model = parts[0]
        number = int(parts[2])
        model_priority = 0 if model == "GNB" else 1
        return (model_priority, number)

    sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro ROC Comparison: Feature Selection')
    plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))
    plt.savefig('micro_roc_fs_comparison.png', bbox_inches='tight')
    plt.close()

    print("\nFeature Selection Results:")
    print("GNB:")
    for n in feature_nums:
        print(f"  FS {n} features - Acc: {gnb_fs_results[n]['acc']:.4f}, Micro AUC: {gnb_fs_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in feature_nums:
        print(f"  FS {n} features - Acc: {mlr_fs_results[n]['acc']:.4f}, Micro AUC: {mlr_fs_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_fs_results[n]['avg_val_f1']:.4f}")

# PCA 比較圖表
def plot_pca_comparison(pca_dims, gnb_pca_results, mlr_pca_results):
    plt.figure(figsize=(12, 8))
    handles = []
    labels = []

    for n in pca_dims:
        line, = plt.plot(gnb_pca_results[n]['fpr_micro'], gnb_pca_results[n]['tpr_micro'], 
                         label=f"GNB PCA {n} ({gnb_pca_results[n]['var_pct'] * 100:.2f}% var, Prec = {gnb_pca_results[n]['prec']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB PCA {n} ({gnb_pca_results[n]['var_pct'] * 100:.2f}% var, Prec = {gnb_pca_results[n]['prec']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")

        line, = plt.plot(mlr_pca_results[n]['fpr_micro'], mlr_pca_results[n]['tpr_micro'], 
                         label=f"MLR PCA {n} ({gnb_pca_results[n]['var_pct'] * 100:.2f}% var, Acc = {mlr_pca_results[n]['prec']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR PCA {n} ({mlr_pca_results[n]['var_pct'] * 100:.2f}% var, Prec = {mlr_pca_results[n]['prec']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")

    def extract_sort_key(label):
        parts = label.split()
        model = parts[0]
        number = int(parts[2])
        model_priority = 0 if model == "GNB" else 1
        return (model_priority, number)

    sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro ROC Comparison: PCA')
    plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))
    plt.savefig('micro_roc_pca_comparison.png', bbox_inches='tight')
    plt.close()

    print("\nPCA Results:")
    print("GNB:")
    for n in pca_dims:
        print(f"  PCA {n} components ({gnb_pca_results[n]['var_pct'] * 100:.2f}% var) - Prec: {gnb_pca_results[n]['prec']:.4f}, Micro AUC: {gnb_pca_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in pca_dims:
        print(f"  PCA {n} components ({mlr_pca_results[n]['var_pct'] * 100:.2f}% var) - Prec: {mlr_pca_results[n]['prec']:.4f}, Micro AUC: {mlr_pca_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_pca_results[n]['avg_val_f1']:.4f}")

def plot_lda_comparison(feature_nums, gnb_lda_results, mlr_lda_results):
    plt.figure(figsize=(12, 8))
    handles = []
    labels = []

    for n in feature_nums:
        # GNB
        line, = plt.plot(gnb_lda_results[n]['fpr_micro'], gnb_lda_results[n]['tpr_micro'],
                         label=f"GNB LDA {n} (Prec = {gnb_lda_results[n]['prec']:.2f}, AUC = {gnb_lda_results[n]['auc_micro']:.2f})")
        labels.append(f"GNB LDA {n} (Prec = {gnb_lda_results[n]['prec']:.2f}, AUC = {gnb_lda_results[n]['auc_micro']:.2f})")
        handles.append(line)

        # MLR
        line, = plt.plot(mlr_lda_results[n]['fpr_micro'], mlr_lda_results[n]['tpr_micro'],
                         label=f"MLR LDA {n} (Prec = {mlr_lda_results[n]['prec']:.2f}, AUC = {mlr_lda_results[n]['auc_micro']:.2f}, F1 = {mlr_lda_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR LDA {n} (Prec = {mlr_lda_results[n]['prec']:.2f}, AUC = {mlr_lda_results[n]['auc_micro']:.2f}, F1 = {mlr_lda_results[n]['avg_val_f1']:.2f})")

    def extract_sort_key(label):
        parts = label.split()
        model = parts[0]
        number = int(parts[2])
        model_priority = 0 if model == "GNB" else 1
        return (model_priority, number)

    sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro ROC Comparison: LDA')
    plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))
    plt.savefig('micro_roc_lda_comparison.png', bbox_inches='tight')
    plt.close()

    print("\nLDA Results:")
    print("GNB:")
    for n in feature_nums:
        print(f"  LDA {n} features - Prec: {gnb_lda_results[n]['prec']:.4f}, Micro AUC: {gnb_lda_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in feature_nums:
        print(f"  LDA {n} features - Prec: {mlr_lda_results[n]['prec']:.4f}, Micro AUC: {mlr_lda_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_lda_results[n]['avg_val_f1']:.4f}")

def plot_pca_fs_roc_comparison(pca_dims, fs_dim, gnb_results, mlr_results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    handles, labels = [], []

    for dim in pca_dims:
        key = (dim, fs_dim)

        # GNB
        if key in gnb_results:
            line, = plt.plot(
                gnb_results[key]['fpr_micro'], gnb_results[key]['tpr_micro'],
                label=f"GNB PCA {dim} (Prec = {gnb_results[key]['prec']:.2f}, AUC = {gnb_results[key]['auc_micro']:.2f})"
            )
            handles.append(line)
            labels.append(f"GNB PCA {dim} (Prec = {gnb_results[key]['prec']:.2f}, AUC = {gnb_results[key]['auc_micro']:.2f})")

        # MLR
        if key in mlr_results:
            line, = plt.plot(
                mlr_results[key]['fpr_micro'], mlr_results[key]['tpr_micro'],
                label=f"MLR PCA {dim} (Prec = {mlr_results[key]['prec']:.2f}, AUC = {mlr_results[key]['auc_micro']:.2f}, F1 = {mlr_results[key]['avg_val_f1']:.2f})"
            )
            handles.append(line)
            labels.append(f"MLR PCA {dim} (Prec = {mlr_results[key]['prec']:.2f}, AUC = {mlr_results[key]['auc_micro']:.2f}, F1 = {mlr_results[key]['avg_val_f1']:.2f})")

    def extract_sort_key(label):
        parts = label.split()
        model = parts[0]
        number = int(parts[2])
        model_priority = 0 if model == "GNB" else 1
        return (model_priority, number)

    sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Micro ROC Comparison: PCA→FS (FS = {fs_dim})')
    plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))
    plt.savefig(f'micro_roc_pca_fs_fs{fs_dim}_comparison.png', bbox_inches='tight')
    plt.close()

    print(f"\nPCA→FS (FS={fs_dim}) Results:")
    print("GNB:")
    for dim in pca_dims:
        if (dim, fs_dim) in gnb_results:
            r = gnb_results[(dim, fs_dim)]
            print(f"  PCA {dim} → FS {fs_dim} - Prec: {r['prec']:.4f}, AUC: {r['auc_micro']:.4f}")
    print("MLR:")
    for dim in pca_dims:
        if (dim, fs_dim) in mlr_results:
            r = mlr_results[(dim, fs_dim)]
            print(f"  PCA {dim} → FS {fs_dim} - Prec: {r['prec']:.4f}, AUC: {r['auc_micro']:.4f}, Avg Val F1: {r['avg_val_f1']:.4f}")


def plot_1d_projection(X_proj, Y, title="1D Projection", filename="1d_projection.png", label_prefix="Component"):
    import matplotlib.pyplot as plt
    Y = np.array(Y)
    X_proj = np.array(X_proj).flatten()  

    plt.figure(figsize=(10, 4))
    classes = np.unique(Y)
    for c in classes:
        plt.scatter(X_proj[Y == c], [c] * np.sum(Y == c), label=f"Class {c}", alpha=0.7, s=20)

    plt.xlabel(f"{label_prefix} 1")
    plt.yticks(classes)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_2d_projection(X_proj, Y, title="2D Projection", filename="2d_projection.png", label_prefix="Component"):
    Y = np.array(Y)
    plt.figure(figsize=(8, 6))
    classes = np.unique(Y)
    for c in classes:
        plt.scatter(X_proj[Y == c, 0], X_proj[Y == c, 1], label=f'Class {c}', alpha=0.7)

    plt.xlabel(f"{label_prefix} 1")
    plt.ylabel(f"{label_prefix} 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3d_projection(X_proj, Y, title="3D Projection", filename="3d_projection.png", label_prefix="Component"):
    Y = np.array(Y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    classes = np.unique(Y)

    for c in classes:
        ax.scatter(X_proj[Y == c, 0], X_proj[Y == c, 1], X_proj[Y == c, 2], label=f'Class {c}', alpha=0.7)

    ax.set_xlabel(f"{label_prefix} 1")
    ax.set_ylabel(f"{label_prefix} 2")
    ax.set_zlabel(f"{label_prefix} 3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_4d_projection(X_proj, Y, title="4D Projection (3D + color)", filename="4d_projection.png", label_prefix="Component"):
    Y = np.array(Y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
        c=X_proj[:, 3], cmap='viridis', alpha=0.7
    )

    ax.set_xlabel(f"{label_prefix} 1")
    ax.set_ylabel(f"{label_prefix} 2")
    ax.set_zlabel(f"{label_prefix} 3")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax, label=f"{label_prefix} 4")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# def plot_pca_fs_comparison(pca_dims, gnb_pca_results, mlr_pca_results, feature_nums, gnb_fs_results, mlr_fs_results):
#     """
#     繪製 PCA 和特徵選取的 Micro ROC 比較圖，並在圖例中顯示 Accuracy 和 Micro AUC
#     """
#     plt.figure(figsize=(16, 8))
#     # 儲存曲線的 handles 和 labels
#     handles = []
#     labels = []

#     # PCA 結果
#     for n in pca_dims:
#         line, = plt.plot(gnb_pca_results[n]['fpr_micro'], gnb_pca_results[n]['tpr_micro'], 
#                          label=f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")
#         handles.append(line)
#         labels.append(f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")

#         line, = plt.plot(mlr_pca_results[n]['fpr_micro'], mlr_pca_results[n]['tpr_micro'], 
#                          label=f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")
#         handles.append(line)
#         labels.append(f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")

#     # 特徵選取結果
#     for n in feature_nums:
#         line, = plt.plot(gnb_fs_results[n]['fpr_micro'], gnb_fs_results[n]['tpr_micro'], 
#                          label=f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
#         handles.append(line)
#         labels.append(f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")

#         line, = plt.plot(mlr_fs_results[n]['fpr_micro'], mlr_fs_results[n]['tpr_micro'], 
#                          label=f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")
#         handles.append(line)
#         labels.append(f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")



#     # 自定義排序：提取模型、方法和數字  
#     def extract_sort_key(label):
#         parts = label.split()
#         model = parts[0]  # GNB 或 MLR
#         method = parts[1]  # FS 或 PCA
#         number = int(parts[2])  # 數字
#         # 定義排序優先級：GNB=0, MLR=1; FS=0, PCA=1; 然後按數字
#         model_priority = 0 if model == "GNB" else 1
#         method_priority = 0 if method == "FS" else 1
#         return (model_priority, method_priority, number)

#     # 按自定義規則排序
#     sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
#     sorted_handles = [handles[i] for i in sorted_indices]
#     sorted_labels = [labels[i] for i in sorted_indices]

#     # 繪製對角線
#     plt.plot([0, 1], [0, 1], 'k--')
#     # 添加排序後的圖例
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Micro ROC Comparison: PCA vs Feature Selection')
#     plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))  # 使用排序後的 handles 和 labels
#     plt.savefig('micro_roc_comparison.png', bbox_inches='tight')  # 確保圖例完整顯示
#     plt.close()
#     # 輸出結果
#     print("\nPCA Results:")
#     print("GNB:")
#     for n in pca_dims:
#         print(f"  {n} components - Acc: {gnb_pca_results[n]['acc']:.4f}, Micro AUC: {gnb_pca_results[n]['auc_micro']:.4f}")
#     print("MLR:")
#     for n in pca_dims:
#         print(f"  {n} components - Acc: {mlr_pca_results[n]['acc']:.4f}, Micro AUC: {mlr_pca_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_pca_results[n]['avg_val_f1']:.4f}")

#     print("\nFeature Selection Results:")
#     print("GNB:")
#     for n in feature_nums:
#         print(f"  {n} features - Acc: {gnb_fs_results[n]['acc']:.4f}, Micro AUC: {gnb_fs_results[n]['auc_micro']:.4f}")
#     print("MLR:")
#     for n in feature_nums:
#         print(f"  {n} features - Acc: {mlr_fs_results[n]['acc']:.4f}, Micro AUC: {mlr_fs_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_fs_results[n]['avg_val_f1']:.4f}")

def predict_and_evaluate(model, X_test, Y_test, model_type):
    y_scores, y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred, len(np.unique(Y_test)))
    plot_confusion_matrix(cm, np.unique(Y_test), f'{model_type} Confusion Matrix', f'{model_type}_confusion_matrix.png')

    acc, prec, rec, f1 = compute_metrics(Y_test, y_pred, len(np.unique(Y_test)))
    print(f'{model_type} - Testing result: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

    fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro = compute_roc_auc(Y_test, y_scores, len(np.unique(Y_test)))
    plot_roc_curve(fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro, len(np.unique(Y_test)), f'{model_type} ROC Curve', f'{model_type}_roc_curve.png')
    return acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro