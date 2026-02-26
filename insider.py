from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

logon_df = pd.read_csv('logon.csv')
device_df = pd.read_csv('device.csv')
email_df = pd.read_csv('email.csv')
file_df = pd.read_csv('file.csv')

print(logon_df.head())
print(device_df.head())
print(email_df.head())
print(file_df.head())

print(logon_df.info())
print(device_df.info())
print(logon_df['user'].value_counts().head())

logon_df['date'] = pd.to_datetime(logon_df['date'], dayfirst=True, format='%m/%d/%Y %H:%M:%S')

activity_counts = logon_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
activity_counts.columns = [f'num_{col.lower()}' for col in activity_counts.columns]

print(activity_counts.head())


def preprocess_log(df, date_col='date'):
    df = df.drop_duplicates(subset='id')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    return df


logon_df = preprocess_log(pd.read_csv('logon.csv'))
device_df = preprocess_log(pd.read_csv('device.csv'))
email_df = preprocess_log(pd.read_csv('email.csv'))
file_df = preprocess_log(pd.read_csv('file.csv'))

# FEATURES

pc_features = logon_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
pc_features.columns = [f'pc_{col.lower()}' for col in pc_features.columns]

logon_df['hour'] = logon_df['date'].dt.hour
logon_df['day_of_week'] = logon_df['date'].dt.dayofweek
logon_df['is_weekend'] = logon_df['day_of_week'].isin([5, 6]).astype(int)

night_logins = logon_df[logon_df['hour'].between(0, 6)]
night_counts = night_logins.groupby('user').size()
pc_features['pc_logins_night'] = night_counts
pc_features['pc_logins_night'] = pc_features['pc_logins_night'].fillna(0)

weekend_logins = logon_df[logon_df['is_weekend'] == 1]
weekend_counts = weekend_logins.groupby('user').size()
pc_features['pc_logins_weekend'] = weekend_counts
pc_features['pc_logins_weekend'] = pc_features['pc_logins_weekend'].fillna(0)

device_features = device_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
device_features.columns = [f'device_{col.lower()}' for col in device_features.columns]

email_df['attachments'] = email_df['attachments'].fillna(0)
email_features = email_df.groupby('user').agg(
    num_emails=('id', 'count'),
)

if 'size' not in file_df.columns:
    file_df['size'] = file_df['content'].apply(lambda x: len(str(x)))
file_df['size'] = file_df['size'].fillna(0)
file_features = file_df.groupby('user').agg(
    total_file_size=('size', 'sum'),
    num_files_downloaded=('id', 'count')
)
file_features['files_per_logon'] = file_features['num_files_downloaded'] / (
            pc_features.get('pc_logon', pd.Series(1)) + 1)

features = pc_features.join(device_features, how='left').fillna(0)
features = features.join(email_features, how='left').fillna(0)
features = features.join(file_features, how='left').fillna(0)

# FEATURES NOVAS
device_cols = [col for col in features.columns if col.startswith('device_')]
features['device_activity_per_logon'] = features[device_cols].sum(axis=1) / (features.get('pc_logon', 1) + 1)

features['files_per_logon'] = features['num_files_downloaded'] / (features.get('pc_logon', 1) + 1)
features['emails_night_ratio'] = features['num_emails'] / (features.get('pc_logins_night', 1) + 1)
device_cols = [col for col in features.columns if col.startswith('device_')]
features['files_per_device_activity'] = features['num_files_downloaded'] / (features[device_cols].sum(axis=1) + 1)

features = features.fillna(0)

print(features.head())
print("\nNúmero de utilizadores:", features.shape[0])
print("Número de features:", features.shape[1])

print(features.dtypes)

# ESTATISTICAS DESCRITIVAS

users = features.index
numeric_cols = features.select_dtypes(include=['number'])

desc_stats = numeric_cols.describe().round(2)
print(desc_stats)

desc_stats.to_csv("estatisticas_descritivas.csv")

corr_matrix = numeric_cols.corr()
print(corr_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matriz de Correlação das Features")
plt.show()

# Análise de distribuições e outliers
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()
for i, col in enumerate(numeric_cols.columns[:12]):
    axes[i].hist(numeric_cols[col], bins=30, edgecolor='black')
    axes[i].set_title(f'Distribuição de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()
for i, col in enumerate(numeric_cols.columns[:12]):
    axes[i].boxplot(numeric_cols[col])
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()

# CLUSTERING
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X = numeric_cols.copy()

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

K_range = range(2, 11)
inertias = []
silhouette_scores = []
silhouette_std = []

from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    fold_scores = []

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train = X_train_scaled[train_idx]
        X_fold_val = X_train_scaled[val_idx]

        kmeans_fold = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_fold = kmeans_fold.fit_predict(X_fold_train)

        if len(np.unique(labels_fold)) > 1:
            score = silhouette_score(X_fold_train, labels_fold)
            fold_scores.append(score)

    if fold_scores:
        silhouette_scores.append(np.mean(fold_scores))
        silhouette_std.append(np.std(fold_scores))

        kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_full = kmeans_full.fit_predict(X_train_scaled)
        inertias.append(kmeans_full.inertia_)
        print(
            f"k={k}: Inércia={kmeans_full.inertia_:.0f}, Silhueta={np.mean(fold_scores):.3f} (+/- {np.std(fold_scores):.3f})")

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Número de clusters")
plt.ylabel("Inércia")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.errorbar(K_range, silhouette_scores, yerr=silhouette_std, marker='o', capsize=5)
plt.title("Silhouette Score vs K (com validação cruzada)")
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print("Melhor k (silhouette com validação cruzada):", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_train = kmeans.fit_predict(X_train_scaled)
cluster_test = kmeans.predict(X_test_scaled)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, cluster_train)

X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_train_df['cluster'] = cluster_train

X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_df['cluster'] = cluster_test

y_test = cluster_test

cluster_summary = X_train_df.groupby('cluster').mean()
cluster_summary['contagem'] = X_train_df.groupby('cluster').size()
print("\n=== Sumário dos Clusters ===")
print(cluster_summary)

# CLUSTERS COM PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nVariância explicada pelos 2 componentes PCA: {pca.explained_variance_ratio_.sum():.2%}")
print(f"Variância por componente: {pca.explained_variance_ratio_}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_train, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Visualização dos Clusters com PCA')
plt.show()

# PARTE SUPERVISIONADA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

models = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced'),
    'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

# Otimização de hiperparâmetros
print("\n OTIMIZAÇÃO DE HIPERPARAMETROS")

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                       param_grid_rf, cv=3, scoring='f1_macro')
grid_rf.fit(X_train_res, y_train_res)
print("Melhores parâmetros Random Forest:", grid_rf.best_params_)
models['Random Forest'] = grid_rf.best_estimator_

# SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1]
}
grid_svm = GridSearchCV(SVC(kernel='rbf', random_state=42, class_weight='balanced'),
                        param_grid_svm, cv=3, scoring='f1_macro')
grid_svm.fit(X_train_res, y_train_res)
print("Melhores parâmetros SVM:", grid_svm.best_params_)
models['SVM'] = grid_svm.best_estimator_

# Decision Tree
param_grid_dt = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                       param_grid_dt, cv=3, scoring='f1_macro')
grid_dt.fit(X_train_res, y_train_res)
print("Melhores parâmetros Decision Tree:", grid_dt.best_params_)
models['Decision Tree'] = grid_dt.best_estimator_

predictions = {}
metrics_list = []

for name, clf in models.items():
    cv_scores = cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring='f1_macro')
    print(f"\n{name} - CV F1-score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test_scaled)
    predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metrics_list.append({
        'Modelo': name,
        'Accuracy': round(acc, 3),
        'Precision (macro)': round(prec, 3),
        'Recall (macro)': round(rec, 3),
        'F1-score (macro)': round(f1, 3),
        'CV F1-score': round(cv_scores.mean(), 3)
    })

comparison_df = pd.DataFrame(metrics_list)
print("\n=== Comparação dos Modelos ===")
print(comparison_df)

rf_clf = models['Random Forest']
feat_imp = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
print("\nTop 10 features do Random Forest:")
print(feat_imp.sort_values(ascending=False).head(10))

plt.figure(figsize=(10, 6))
feat_imp.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Features mais Importantes (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importância')
plt.tight_layout()
plt.show()

# Random Forest
rf_clf = models['Random Forest']
y_pred_rf = rf_clf.predict(X_test_scaled)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Logistic Regression
lr = models['Logistic Regression']
y_pred_lr = lr.predict(X_test_scaled)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# SVM
svm_clf = models['SVM']
y_pred_svm = svm_clf.predict(X_test_scaled)
print("\n=== SVM ===")
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Decision Tree
dt_clf = models['Decision Tree']
y_pred_dt = dt_clf.predict(X_test_scaled)
print("\n=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

from sklearn.tree import plot_tree
from sklearn.tree import export_text

plt.figure(figsize=(30, 15))
plot_tree(dt_clf,
          feature_names=X_train.columns,
          class_names=[str(i) for i in range(best_k)],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Árvore de Decisão Completa')
plt.tight_layout()
plt.show()

# Análise de overfitting
print("\n=== Análise de Overfitting ===")
for name, clf in models.items():
    train_score = clf.score(X_train_res, y_train_res)
    test_score = clf.score(X_test_scaled, y_test)
    gap = train_score - test_score
    print(f"{name}: Treino={train_score:.3f}, Teste={test_score:.3f}, Gap={gap:.3f}")
    if gap > 0.1:
        print(f" Possível overfitting em {name}")
    elif gap < -0.05:
        print(f" Possível underfitting em {name}")
    else:
        print(f"  ✓ Modelo bem ajustado")