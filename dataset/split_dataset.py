import pandas as pd

SEED = 42 # semente para caso eu precise embaralhar dnv com a mesma divisão

df = pd.read_csv("student-por.csv")

# Embaralhar instâncias
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Divisão 80/20
train_size = int(0.8 * len(df_shuffled))
train_df = df_shuffled.iloc[:train_size]
test_df = df_shuffled.iloc[train_size:]

# Salvar em arquivos
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)