import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar dataset
df = pd.read_csv("dataset/student-por.csv")

# Selecionar apenas colunas numéricas
num_features = df.select_dtypes(include=['int64', 'float64'])

# Matriz de correlação Pearson completa
corr_matrix = num_features.corr()

# Criar máscara para somente a parte acima da diagonal (não inclui diagonal)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot do heatmap
plt.figure(figsize=(12,6))  # tamanho adequado
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação Pearson - Variáveis Numéricas")
plt.xticks(rotation=45, ha='right')  # rotaciona labels X
plt.yticks(rotation=0)  # mantém labels Y horizontais
plt.tight_layout()  # ajusta espaçamento para caber tudo
plt.show()

