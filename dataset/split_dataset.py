import pandas as pd

SEED = 42 # Guardo a semente do shuffle

# Carregar dataset
df = pd.read_csv("dataset/student-por.csv")

# Mapear variáveis binárias 'yes'/'no' para 1/0
binarias = ["higher", "internet", "romantic"]
df[binarias] = df[binarias].applymap(lambda x: 1 if x == "yes" else 0)

# Selecionar atributos e target
cols = ["G2", "G1", "failures", "studytime", "Medu", "Fedu", 
        "higher", "internet", "romantic", "G3"]
df = df[cols]

# Embaralhar instâncias
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Divisão 80/20
train_size = int(0.8 * len(df_shuffled))
train_df = df_shuffled.iloc[:train_size]
test_df = df_shuffled.iloc[train_size:]

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)