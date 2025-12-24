import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

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

    X_test, Y_test = X[test_indices], Y[test_indices]
    train_val_split = k_fold_cross_validation(len(train_val_indices), k=k, shuffle=True, random_seed=random_seed)
    
    print("Class distribution in full dataset:", np.bincount(Y))
    print("Class distribution in test set:", np.bincount(Y_test))
    print("Class distribution in train/val set:", np.bincount(Y[train_val_indices]))

    return X_test, Y_test, train_val_indices, train_val_split

def compute_pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, :n_components]
    X_pca = np.dot(X_centered, W)

    # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # sorted_eigenvalues = S**2 / (X.shape[0] - 1)
    # W = Vt.T[:, :n_components]
    # X_pca = U * S

    return X_pca, W, sorted_eigenvalues

def compute_mutual_information(X, Y):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(Y))
    mi = np.zeros(n_features)

    for f in range(n_features):
        X_feature = X[:, f]
        bins = np.linspace(np.min(X_feature), np.max(X_feature), 16)
        X_feature_binned = np.digitize(X_feature, bins) - 1
        X_feature_binned = np.clip(X_feature_binned, 0, len(bins) - 2)  # 限制範圍為 0 到 14

        joint_hist = np.zeros((n_classes, len(bins) - 1))  # 形狀為 (n_classes, 15)
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
                         label=f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
        line, = plt.plot(mlr_fs_results[n]['fpr_micro'], mlr_fs_results[n]['tpr_micro'], 
                         label=f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")

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
                         label=f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")
        line, = plt.plot(mlr_pca_results[n]['fpr_micro'], mlr_pca_results[n]['tpr_micro'], 
                         label=f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")

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
        print(f"  PCA {n} components - Acc: {gnb_pca_results[n]['acc']:.4f}, Micro AUC: {gnb_pca_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in pca_dims:
        print(f"  PCA {n} components - Acc: {mlr_pca_results[n]['acc']:.4f}, Micro AUC: {mlr_pca_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_pca_results[n]['avg_val_f1']:.4f}")
        
def plot_pca_fs_comparison(pca_dims, gnb_pca_results, mlr_pca_results, feature_nums, gnb_fs_results, mlr_fs_results):
    """
    繪製 PCA 和特徵選取的 Micro ROC 比較圖，並在圖例中顯示 Accuracy 和 Micro AUC
    """
    plt.figure(figsize=(16, 8))
    # 儲存曲線的 handles 和 labels
    handles = []
    labels = []

    # PCA 結果
    for n in pca_dims:
        line, = plt.plot(gnb_pca_results[n]['fpr_micro'], gnb_pca_results[n]['tpr_micro'], 
                         label=f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB PCA {n} (Acc = {gnb_pca_results[n]['acc']:.2f}, AUC = {gnb_pca_results[n]['auc_micro']:.2f})")

        line, = plt.plot(mlr_pca_results[n]['fpr_micro'], mlr_pca_results[n]['tpr_micro'], 
                         label=f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR PCA {n} (Acc = {mlr_pca_results[n]['acc']:.2f}, AUC = {mlr_pca_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_pca_results[n]['avg_val_f1']:.2f})")

    # 特徵選取結果
    for n in feature_nums:
        line, = plt.plot(gnb_fs_results[n]['fpr_micro'], gnb_fs_results[n]['tpr_micro'], 
                         label=f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")
        handles.append(line)
        labels.append(f"GNB FS {n} (Acc = {gnb_fs_results[n]['acc']:.2f}, AUC = {gnb_fs_results[n]['auc_micro']:.2f})")

        line, = plt.plot(mlr_fs_results[n]['fpr_micro'], mlr_fs_results[n]['tpr_micro'], 
                         label=f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")
        handles.append(line)
        labels.append(f"MLR FS {n} (Acc = {mlr_fs_results[n]['acc']:.2f}, AUC = {mlr_fs_results[n]['auc_micro']:.2f}, Avg_val_f1 = {mlr_fs_results[n]['avg_val_f1']:.2f})")



    # 自定義排序：提取模型、方法和數字  
    def extract_sort_key(label):
        parts = label.split()
        model = parts[0]  # GNB 或 MLR
        method = parts[1]  # FS 或 PCA
        number = int(parts[2])  # 數字
        # 定義排序優先級：GNB=0, MLR=1; FS=0, PCA=1; 然後按數字
        model_priority = 0 if model == "GNB" else 1
        method_priority = 0 if method == "FS" else 1
        return (model_priority, method_priority, number)

    # 按自定義規則排序
    sorted_indices = sorted(range(len(labels)), key=lambda i: extract_sort_key(labels[i]))
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # 繪製對角線
    plt.plot([0, 1], [0, 1], 'k--')
    # 添加排序後的圖例
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro ROC Comparison: PCA vs Feature Selection')
    plt.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.4, 0))  # 使用排序後的 handles 和 labels
    plt.savefig('micro_roc_comparison.png', bbox_inches='tight')  # 確保圖例完整顯示
    plt.close()
    # 輸出結果
    print("\nPCA Results:")
    print("GNB:")
    for n in pca_dims:
        print(f"  {n} components - Acc: {gnb_pca_results[n]['acc']:.4f}, Micro AUC: {gnb_pca_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in pca_dims:
        print(f"  {n} components - Acc: {mlr_pca_results[n]['acc']:.4f}, Micro AUC: {mlr_pca_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_pca_results[n]['avg_val_f1']:.4f}")

    print("\nFeature Selection Results:")
    print("GNB:")
    for n in feature_nums:
        print(f"  {n} features - Acc: {gnb_fs_results[n]['acc']:.4f}, Micro AUC: {gnb_fs_results[n]['auc_micro']:.4f}")
    print("MLR:")
    for n in feature_nums:
        print(f"  {n} features - Acc: {mlr_fs_results[n]['acc']:.4f}, Micro AUC: {mlr_fs_results[n]['auc_micro']:.4f}, Avg Val F1: {mlr_fs_results[n]['avg_val_f1']:.4f}")

def predict_and_evaluate(model, X_test, Y_test, model_type):
    y_scores, y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred, len(np.unique(Y_test)))
    plot_confusion_matrix(cm, np.unique(Y_test), f'{model_type} Confusion Matrix', f'{model_type}_confusion_matrix.png')

    acc, prec, rec, f1 = compute_metrics(Y_test, y_pred, len(np.unique(Y_test)))
    print(f'{model_type} - Testing result: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

    fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro = compute_roc_auc(Y_test, y_scores, len(np.unique(Y_test)))
    plot_roc_curve(fpr, tpr, auc, fpr_micro, tpr_micro, auc_micro, len(np.unique(Y_test)), f'{model_type} ROC Curve', f'{model_type}_roc_curve.png')
    return acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro