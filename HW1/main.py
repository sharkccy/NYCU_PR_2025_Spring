import numpy as np
import pickle
import mutils as mutils

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, Y):
        n_sample, n_features = X.shape
        self.classes = np.unique(Y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[Y == c]
            self.means[idx] = np.mean(X_c, axis=0)
            self.variances[idx] = np.var(X_c, axis=0)
            self.priors[idx] = len(X_c) / n_sample

    def predict(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * self.variances[idx]) + \
                        ((X - self.means[idx]) ** 2) / (2 * self.variances[idx]), axis=1)
            log_probs[:, idx] = np.log(self.priors[idx]) + log_prob

        probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs, np.argmax(log_probs, axis=1)

class MultinomialLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))
        
        self.weights = np.zeros((n_features, n_classes))
        y_one_hot = np.eye(n_classes)[Y]

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights)
            probs = self.softmax(z)
            error = probs - y_one_hot
            gradient = np.dot(X.T, error) / n_samples
            self.weights -= self.lr * gradient
            self.lr = self.lr * 0.95
            # if epoch % 100 == 0:
            #     loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-10), axis=1))
            #     print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        z = np.dot(X, self.weights)
        probs = self.softmax(z)
        return probs, np.argmax(probs, axis=1)
    
def train_model(X_train, Y_train, X_val, Y_val, model_type='mlr'):
    if model_type == 'mlr':
        model = MultinomialLogisticRegression(lr=1e-2, epochs=1000)
    else:
        raise ValueError("Unknown model type. Choose 'gnb' or 'mlr'.")
    
    model.fit(X_train, Y_train)
    
    _, Y_train_pred = model.predict(X_train)
    # cm_train = mutils.confusion_matrix(Y_train, Y_train_pred, len(np.unique(Y_train)))
    acc_train, prec_train, rec_train, f1_train = mutils.compute_metrics(Y_train, Y_train_pred, len(np.unique(Y_train)))
    
    _, Y_val_pred = model.predict(X_val)
    # cm_val = mutils.confusion_matrix(Y_val, Y_val_pred, len(np.unique(Y_val)))
    acc_val, prec_val, rec_val, f1_val = mutils.compute_metrics(Y_val, Y_val_pred, len(np.unique(Y_train))) #use Y_train to prevent there's a type missing in Y_val

    if model_type == 'mlr':
        probs_train = model.softmax(np.dot(X_train, model.weights))
        probs_val = model.softmax(np.dot(X_val, model.weights))
        Y_train_one_hot = np.eye(len(np.unique(Y_train)))[Y_train]
        Y_val_one_hot = np.eye(len(np.unique(Y_train)))[Y_val]  # Y_val is used to prevent there's a type missing in Y_val
        # Y_val_one_hot = np.eye(len(np.unique(Y_val)))[Y_val]
        loss_train = -np.mean(np.sum(Y_train_one_hot * np.log(probs_train), axis=1))
        loss_val = -np.mean(np.sum(Y_val_one_hot * np.log(probs_val), axis=1))
    else:
        loss_train = 1 - acc_train
        loss_val = 1 - acc_val
    
    return {
        'train_acc' : acc_train,
        'val_acc' : acc_val,
        'train_loss' : loss_train,
        'val_loss' : loss_val,
        'train_f1' : np.mean(f1_train),
        'val_f1' : np.mean(f1_val),
    }, model


save_path = 'models/'
data_paths = ['data/obesity.csv', 'data/wine.csv', 'data/forest.csv', 'data/churn.csv']
data_path = data_paths[3]

X, Y, _ = mutils.load_and_preprocess_data(data_path)
X_test, Y_test, train_val_indices, train_val_split = mutils.split_data(X, Y, k=10, shuffle=True, random_seed=42)
X_train_val, Y_train_val = X[train_val_indices], Y[train_val_indices]

n_features = X.shape[1]
print(f"Original number of features: {n_features}")

pca_dims = [n_features, int(n_features // 1.5), n_features // 2, n_features // 3]
gnb_pca_results = {}
mlr_pca_results = {}


for n_components in pca_dims:
        print(f"\nRunning PCA with {n_components} components...")
        X_pca, W, eigenvalues = mutils.compute_pca(X, n_components=n_components)
        print(f"Explained variance ratio for {n_components} components:", np.sum(eigenvalues[:n_components] / np.sum(eigenvalues)))

        X_pca_test = ((X_test - np.mean(X, axis=0)) / np.std(X, axis=0)) @ W
        X_pca_train_val = X_pca[train_val_indices]

        # GNB with PCA
        gnb_model = GaussianNaiveBayes()
        gnb_model.fit(X_pca_train_val, Y_train_val)
        acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro = mutils.predict_and_evaluate(gnb_model, X_pca_test, Y_test, model_type=f'gnb_pca_{n_components}')
        gnb_pca_results[n_components] = {'acc': acc, 'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro}

        # MLR with PCA
        mlr_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        for fold, (train_idx, val_idx) in enumerate(train_val_split):
            X_train, Y_train = X_pca_train_val[train_idx], Y_train_val[train_idx]
            X_val, Y_val = X_pca_train_val[val_idx], Y_train_val[val_idx]
            mlr_metrics, mlr_model = train_model(X_train, Y_train, X_val, Y_val, model_type='mlr')
            for key in mlr_history:
                mlr_history[key].append(mlr_metrics[key])

        avg_val_f1 = np.mean(mlr_history['val_f1'])
        acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro  = mutils.predict_and_evaluate(mlr_model, X_pca_test, Y_test, model_type=f'mlr_pca_{n_components}')
        mlr_pca_results[n_components] = {'acc': acc, 'avg_val_f1' : avg_val_f1,'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro}
        mutils.plot_metrics(mlr_history, f'MLR PCA {n_components}', f'mlr_pca_{n_components}_metrics.png')

# 特徵選取實驗
feature_nums = [n_features, int(n_features // 1.5), n_features // 2, n_features // 3]
gnb_fs_results = {}
mlr_fs_results = {}

for n_features_selected in feature_nums:
    print(f"\nRunning Feature Selection with {n_features_selected} features...")
    X_selected, selected_indices = mutils.select_features(X, Y, n_features_selected)
    X_selected_test = X_test[:, selected_indices]
    X_selected_train_val = X_selected[train_val_indices]

    # GNB with Feature Selection
    gnb_model = GaussianNaiveBayes()
    gnb_model.fit(X_selected_train_val, Y_train_val)
    acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro = mutils.predict_and_evaluate(gnb_model, X_selected_test, Y_test, model_type=f'gnb_fs_{n_features_selected}')
    gnb_fs_results[n_features_selected] = {'acc': acc, 'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro}

    # MLR with Feature Selection
    mlr_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    for fold, (train_idx, val_idx) in enumerate(train_val_split):
        X_train, Y_train = X_selected_train_val[train_idx], Y_train_val[train_idx]
        X_val, Y_val = X_selected_train_val[val_idx], Y_train_val[val_idx]
        mlr_metrics, mlr_model = train_model(X_train, Y_train, X_val, Y_val, model_type='mlr')
        for key in mlr_history:
            mlr_history[key].append(mlr_metrics[key])
    avg_val_f1 = np.mean(mlr_history['val_f1'])
    acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro = mutils.predict_and_evaluate(mlr_model, X_selected_test, Y_test, model_type=f'mlr_fs_{n_features_selected}')
    mlr_fs_results[n_features_selected] = {'acc': acc, 'avg_val_f1' : avg_val_f1,'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro}
    mutils.plot_metrics(mlr_history, f'MLR FS {n_features_selected}', f'mlr_fs_{n_features_selected}_metrics.png')

# 原始數據的 GNB 和 MLR
print("\nRunning original data models...")
gnb_model = GaussianNaiveBayes()
gnb_model.fit(X_train_val, Y_train_val)
with open(save_path + 'gnb_model.pkl', 'wb') as f:
    pickle.dump(gnb_model, f)
print("Gaussian Naive Bayes model saved.")
acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro = mutils.predict_and_evaluate(gnb_model, X_test, Y_test, model_type='gnb')
gnb_pca_results[n_features] = {'acc': acc, 'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro} # 更新原始數據結果

mlr_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
for fold, (train_idx, val_idx) in enumerate(train_val_split):
    print(f"Fold {fold + 1}/{len(train_val_split)} - Training...")
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    mlr_metrics, mlr_model = train_model(X_train, Y_train, X_val, Y_val, model_type='mlr')
    for key in mlr_history:
        mlr_history[key].append(mlr_metrics[key])
    print(f"Fold {fold + 1}/{len(train_val_split)} - loss: {mlr_metrics['train_loss']:.4f} ")

avg_val_f1 = np.mean(mlr_history['val_f1'])

acc, prec, rec, f1, fpr_micro, tpr_micro, auc_micro = mutils.predict_and_evaluate(mlr_model, X_test, Y_test, model_type='mlr')
mlr_pca_results[n_features] = {'acc': acc, 'avg_val_f1' : avg_val_f1,'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro, 'auc_micro': auc_micro} # 更新原始數據結果
mutils.plot_metrics(mlr_history, 'Multinomial Logistic Regression', 'mlr_metrics.png')

with open(save_path + 'mlr_model.pkl', 'wb') as f:
    pickle.dump(mlr_model, f)
print("Multinomial Logistic Regression model saved.")

# 繪製比較圖並輸出結果
# mutils.plot_pca_fs_comparison(pca_dims, gnb_pca_results, mlr_pca_results, feature_nums, gnb_fs_results, mlr_fs_results)
mutils.plot_fs_comparison(feature_nums, gnb_fs_results, mlr_fs_results)
mutils.plot_pca_comparison(pca_dims, gnb_pca_results, mlr_pca_results)
print("Training and evaluation completed.")

