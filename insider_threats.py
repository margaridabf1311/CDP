from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


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
    num_emails_with_attachments=('attachments', 'sum')
)
email_features['ratio_emails_with_attachments'] = email_features['num_emails_with_attachments'] / (email_features['num_emails'] + 1)


if 'size' not in file_df.columns:
    file_df['size'] = file_df['content'].apply(lambda x: len(str(x)))
file_df['size'] = file_df['size'].fillna(0)
file_features = file_df.groupby('user').agg(
    num_files_downloaded=('id', 'count'),
    total_file_size=('size', 'sum')
)
file_features['files_per_logon'] = file_features['num_files_downloaded'] / (pc_features.get('pc_logon', pd.Series(1)) + 1)

features = pc_features.join(device_features, how='left').fillna(0)
features = features.join(email_features, how='left').fillna(0)
features = features.join(file_features, how='left').fillna(0)

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

#FALTA ESCOLHER O NUMERO DE CLUSTERS!!!


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

features['cluster'] = clusters

print(features['cluster'].value_counts())

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
    'cluster': clusters,
    'user': features.index
})

var_ratio = pca.explained_variance_ratio_
print(f"\n   ✅ Variância explicada:")
print(f"      PC1: {var_ratio[0]:.1%}")
print(f"      PC2: {var_ratio[1]:.1%}")
print(f"      Total: {var_ratio[0]+var_ratio[1]:.1%} da variância explicada")

plt.figure(figsize=(12, 8))

cores = ['blue', 'green', 'red']
nomes_clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']

for cluster in range(3):
    mask = pca_df['cluster'] == cluster
    plt.scatter(pca_df.loc[mask, 'PC1'], 
                pca_df.loc[mask, 'PC2'],
                c=cores[cluster], 
                label=nomes_clusters[cluster],
                alpha=0.6, 
                edgecolors='black', 
                linewidth=0.5,
                s=100)

plt.xlabel(f'PC1 ({var_ratio[0]:.1%} variância)', fontsize=12)
plt.ylabel(f'PC2 ({var_ratio[1]:.1%} variância)', fontsize=12)
plt.title('Visualização dos Clusters com PCA', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
plt.show()



"""
features['outlier'] = 0
for c in features['cluster'].unique():
    idx = features['cluster'] == c
    X_cluster = numeric_cols.loc[idx]
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X_cluster_scaled)
    features.loc[idx, 'outlier'] = preds

features['suspect'] = (features['outlier'] == -1).astype(int)
print(features['suspect'].value_counts())

features.to_csv("insider_threat_final.csv")

suspects = features[features['suspect'] == 1]
print(suspects.sort_values(by=['files_per_logon', 'pc_logins_night'], ascending=False))
"""