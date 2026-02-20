import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

features=pd.read_csv('insider_threat_clusters.csv')

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X = features.drop(columns=['cluster'])
X=X.select_dtypes(include=[np.number])
y = features['cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# scaler supervisionado (NOVO)
scaler_sup = StandardScaler()
X_train_scaled = scaler_sup.fit_transform(X_train)
X_test_scaled = scaler_sup.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#RANDOM FOREST
clf = RandomForestClassifier(random_state=42)
y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#LOGISTIC REGRESSION

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000,solver='lbfgs')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))

#SVM

svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)

print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

#ARVORE DECISAO
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf_cv = RandomForestClassifier(random_state=42)

scores = cross_val_score(
    rf_cv,
    X_train,
    y_train,
    cv=5,
    scoring='f1_macro'
)

print("CV F1 Macro:", scores)
print("Mean:", scores.mean())

feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=False).head(10)
print(feat_imp)


# Calcular a matriz de correlação
corr_matrix = X.join(y).corr()

# Plotar heatmap
plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    center=0,            # destaca correlações negativas vs positivas
    linewidths=0.5,      # linhas entre células
    cbar_kws={'shrink':0.8}  # barra de cores menor
)
plt.title("Matriz de Correlação das Features e Cluster", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()