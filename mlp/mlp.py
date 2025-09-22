import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ==================== TREINO ====================
ARQUIVO_TREINO = "dataset/train.csv"
ARQUIVO_TESTE = "dataset/test.csv"

# Carregar treino
dados_planilha = np.loadtxt(ARQUIVO_TREINO, delimiter=',', dtype=float, skiprows=1)

# Última coluna (G3) como var dependente
var_dependentes = dados_planilha[:, -1]  
# Todas as outras como independentes
var_independentes = dados_planilha[:, :-1]  

# ==================== TESTE ====================
dados_planilha = np.loadtxt(ARQUIVO_TESTE, delimiter=',', dtype=float, skiprows=1)

var_dep = dados_planilha[:, -1]   # Y
var_ind = dados_planilha[:, :-1]  # X

# ==================== MODELO MLP ====================
# MLPRegressor = rede neural para regressão
modelo = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # duas camadas ocultas: 64 e 32 neurônios
    activation='relu',            # função de ativação
    solver='adam',                # otimizador
    max_iter=500,                 # número de épocas
    random_state=42
)

# Treinar modelo
modelo.fit(var_independentes, var_dependentes)

# Fazer previsões no teste
y_pred = modelo.predict(var_ind)

# Avaliar desempenho
mse = mean_squared_error(var_dep, y_pred)
print(f"Mean Squared Error (MSE) no teste: {mse:.4f}")
