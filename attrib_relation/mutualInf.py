import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dataset
df = pd.read_csv("dataset/student-por.csv")
y = df['G3']

# Mapear binárias 'yes'/'no' para 1/0
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() == 2:
        df[col] = df[col].map({'yes':1, 'no':0})

# Selecionar apenas colunas categóricas e binárias
cat_bin_features = [col for col in df.columns if df[col].dtype == 'object' or (df[col].nunique() == 2 and col != 'G3')]

# Codificar categóricas em dummies
X = pd.get_dummies(df[cat_bin_features], drop_first=True)

# Calcular Mutual Information
mi_scores = mutual_info_regression(X, y, discrete_features=True)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Visualizar
plt.figure(figsize=(12,20))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.xlabel("Mutual Information com G3")
plt.ylabel("Atributos")
plt.title("Relevância de Variáveis Binárias e Categóricas para G3")
plt.show()
