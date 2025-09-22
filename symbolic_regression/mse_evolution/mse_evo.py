from pysr import PySRRegressor  
import numpy as np
from sklearn.metrics import mean_squared_error

ARQUIVO_TREINO = "dataset/train.csv"
ARQUIVO_TESTE = "dataset/test.csv"

# Carrego toda a planilha
dados_planilha = np.loadtxt(ARQUIVO_TREINO, delimiter=',', dtype=float, skiprows=1)

# Última coluna (G3) como var dependente
var_dependentes = dados_planilha[:, -1]  

# Todas as outras como vars independentes
var_independentes = dados_planilha[:, :-1]  


# Carrego os dados da planilha para a memória ==================
dados_planilha = np.loadtxt(ARQUIVO_TESTE, delimiter=',', dtype=float, skiprows=1)

# Faço a divisão entre as variáveis dependentes e independentes 
var_dep = dados_planilha[ : , -1] # Contém somente Y
var_ind = dados_planilha[:, :-1]    # Contém somente X                        

# Definindo meu modelo ===================
modelo = PySRRegressor(
    model_selection="best",
    niterations=1,  # 1 para ficar fiel ao numero de iterações
    populations=1, # Default = 15
    population_size=200, # Default = 33
    maxsize=30, # Limita a complexidade máxima das equações (Default = 20)
    #ncycles_per_iteration=1100,
    binary_operators=["+", "*", "-", "/", "^"], # Defino os operadores binários que vou usar (já tem os mais importantes)
    unary_operators=[ # Defino os operadores unários que vou usar (já tem os mais importantes)
        "sin",
        "cos",
        "exp",
        "log",
        "sinh",
        "cosh",
        "erf",
    ],
    turbo=True, # Tende a acelerar o processo de treinamento, mas pode gerar erros em alguns casos
    bumper=True, # Mesmo funcionamento que o turbo
    #batching=True,
    warm_start=True, # Deixar sempre true para ter a possibilidade de continuar de onde parou
    nested_constraints={"sin": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "cos": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1},
                        "sinh": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "cosh": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1},
                        "erf": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "log": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}},
    constraints={"^": (9, 1)},
    annealing=True,
    progress=False, # Desativo a barra de progresso (Eu acho feio ¯\_(ツ)_/¯)
    verbosity=1, # Defina como 1 para mostrar todas as equações na hora do treinamento
    run_id="result_mse_evo", # Caminho para onde vai o modelo treinado
)
modelo.tournament_selection_n = int(modelo.population_size*0.303) # Default = 15 /tamanho do torneio
modelo.topn = int(modelo.population_size*0.303) # Default = 12 / elitismo

iteracoes = modelo.niterations

#modelo.fit(X=var_independentes, y=var_dependentes) pra testar

for i in range(1,7):
    modelo.fit(X=var_independentes, y=var_dependentes)
    
    with open("symbolic_regression/mse_evolution/dados.txt", "a") as f:
        f.write(f"{iteracoes},{mean_squared_error(var_dep, modelo.predict(var_ind))}\n")
    modelo.niterations = 10**i - iteracoes
    iteracoes = 10**i

print(modelo)
