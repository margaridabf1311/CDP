from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification


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

logon_df['date'] = pd.to_datetime(logon_df['date'],dayfirst=True, format='%m/%d/%Y %H:%M:%S')

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


#FEATURES

pc_features = logon_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
pc_features.columns = [f'pc_{col.lower()}' for col in pc_features.columns]


logon_df['hour'] = logon_df['date'].dt.hour
night_logins = logon_df[logon_df['hour'].between(0,6)]
night_counts = night_logins.groupby('user').size()

pc_features['pc_logins_night'] = night_counts
pc_features['pc_logins_night'] = pc_features['pc_logins_night'].fillna(0)


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
file_features['files_per_logon'] = file_features['num_files_downloaded'] / (pc_features.get('pc_logon', pd.Series(1)) + 1)

features = pc_features.join(device_features, how='left').fillna(0)
features = features.join(email_features, how='left').fillna(0)
features = features.join(file_features, how='left').fillna(0)


#FEATURES NOVAS
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

#ESTATISTICAS DESCRITIVAS

users = features.index  
numeric_cols = features.select_dtypes(include=['number'])

desc_stats = numeric_cols.describe().round(2)
print(desc_stats)

desc_stats.to_csv("estatisticas_descritivas.csv")

corr_matrix = numeric_cols.corr()
print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matriz de Correlação das Features")
plt.show()

#CLUSTERING
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X = numeric_cols.copy()

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

K_range = range(2, 11)  # Testar de 2 a 10 clusters
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    try:
        labels = kmeans.fit_predict(X_train_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_train_scaled, labels)
        silhouette_scores.append(sil_score)
        print(f"k={k}: Inércia={kmeans.inertia_:.0f}, Silhueta={sil_score:.3f}")
    except ValueError as e:
        print(f"Erro ao calcular k={k}: {e}")
        inertias.append(np.nan)
        silhouette_scores.append(np.nan)

plt.figure(figsize=(10,5))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Número de clusters")
plt.ylabel("Inércia")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print("Melhor k (silhouette):", best_k)


kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_train = kmeans.fit_predict(X_train_scaled)
cluster_test = kmeans.predict(X_test_scaled)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, cluster_train)


X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_train_df['cluster'] = cluster_train

X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_df['cluster'] = cluster_test

y_test=cluster_test

cluster_summary = X_train_df.groupby('cluster').mean()
cluster_summary['contagem'] = X_train_df.groupby('cluster').size()
print(cluster_summary)

#CLUSTERS COM PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#PARTE SUPERVISIONADA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


models = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced'),
    'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

predictions = {}
metrics_list = []

for name, clf in models.items():
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
        'F1-score (macro)': round(f1, 3)
    })


comparison_df = pd.DataFrame(metrics_list)
print("=== Comparação dos Modelos com Teste Artificial ===")
print(comparison_df)


rf_clf = models['Random Forest']
feat_imp = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
print("\nTop 10 features do Random Forest:")
print(feat_imp.sort_values(ascending=False).head(10))


# Random Forest
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_clf.fit(X_train_res, y_train_res)
y_pred_rf = rf_clf.predict(X_test_scaled)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Logistic Regression
lr = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test_scaled)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

# SVM
svm_clf = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_clf.fit(X_train_res, y_train_res)
y_pred_svm = svm_clf.predict(X_test_scaled)
print("\n=== SVM ===")
print(classification_report(y_test, y_pred_svm))

# Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_clf.fit(X_train_res, y_train_res)
y_pred_dt = dt_clf.predict(X_test_scaled)
print("\n=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))