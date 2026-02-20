from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split


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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_cols)

K_range = range(2, 11)  # Testar de 2 a 10 clusters
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in K_range:
    print(f"   Testando k={k}...", end=" ")
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calcular métricas
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    
    print(f"Inércia: {kmeans.inertia_:.0f}, Silhueta: {sil_score:.3f}")

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

from sklearn.metrics import adjusted_rand_score

# dividir dados
X_train_u, X_test_u = train_test_split(X_scaled, test_size=0.2, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

features['cluster'] = clusters

cluster_summary = numeric_cols.groupby(clusters).mean().round(2)
cluster_summary['contagem'] = features.groupby('cluster').size()
print(cluster_summary)

features.to_csv("insider_threat_clusters.csv")

#CLUSTERS COM PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'cluster': features['cluster'],
    'user': features.index
})

plt.figure(figsize=(10,8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='tab10')
plt.title('Clusters (PCA)')
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=300)
plt.show()