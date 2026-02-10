import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore


df_ml=pd.read_csv("tsp_features_heuristics_results.csv")

print("\nResumo estatístico das features:\n")
print(df_ml.describe())


plt.figure(figsize=(6,4))
sns.countplot(data=df_ml, x='best_heuristic', order=df_ml['best_heuristic'].value_counts().index)
plt.title("Distribuição da melhor heurística")
plt.show()


features_to_plot = ['num_cities', 'hull_area_ratio', 'mst_ratio', 'coord_spread_ratio']
plt.figure(figsize=(12,6))
sns.boxplot(data=df_ml[features_to_plot])
plt.title("Boxplots das features selecionadas")
plt.show()


numeric_cols = df_ml.select_dtypes(include=np.number).columns.tolist()
z_scores = np.abs(zscore(df_ml[numeric_cols]))
outliers = (z_scores > 3).sum(axis=0)
print("\nNúmero de outliers por feature (Z>3):\n", outliers)


le = LabelEncoder()
df_ml['best_heuristic_encoded'] = le.fit_transform(df_ml['best_heuristic'])
print("\nEncoding das heurísticas:", dict(zip(le.classes_, le.transform(le.classes_))))


X_numeric = df_ml[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_ml['PCA1'] = X_pca[:,0]
df_ml['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_ml, x='PCA1', y='PCA2', hue='best_heuristic', palette='tab10')
plt.title("PCA das features colorido pela melhor heurística")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df_ml['cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_ml, x='PCA1', y='PCA2', hue='cluster', palette='Set2')
plt.title("Clusters das instâncias (PCA 2D)")
plt.show()

# --- 7️⃣ Correlações ---
plt.figure(figsize=(12,10))
corr = df_ml[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlação das features")
plt.show()