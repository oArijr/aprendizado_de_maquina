def print_title(title):
    print(f"\n{'=' * 60}")
    print(f"{title}:")
    print(f"{'=' * 60}")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV


"""
#PARTE 4: SVM

1. Aplique o algoritmo SVM na base de dados ‘credit.pkl’.
2. Inicialmente treine o SVM com kernel linear, valor do parâmetro C = 1.0 e ‘random_state =1’
3. Utilize o comando do sklearn accuray_score para calcular a acurácia do seu algoritmo. O resultado deve ser 0.946
4. Teste os demais kernels e anote os resultados. Qual o melhor kernel para a sua base de dados?
    * Polinomial
    * Sigmoide
    * rbf
5. Aumente o valor do parâmetro C aplicado ao melhor kernel e verifique se há mudanças no resultado do seu SVM.
6. O Grid Search (pesquisa em grade) é uma técnica utilizada para melhorar a precisão e a generalização dos modelos de aprendizado de máquina. 
Ela é usada para realizar ajustes de hiperparâmetros durante o treinamento de um modelo. O grid search automatiza o processo de encontrar hiperparâmetros ideais, economizando esforço humano em comparação com o ajuste manual, mas pode até ser mais custoso do ponto de vista de desempenho, pois testa todas as combinações possíveis e retorna a que obteve melhor desempenho.
Agora, aplique o GridSearch do Scikit-Learn (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) para ajudar a identificar os melhores hiperparâmetros para o seu modelo. Qual foi a melhor combinação de hiperparâmetros encontrada? O modelo com melhor desempenho foi obtido com os parâmetros ajustados manualmente ou com o GridSearch?
"""
with open("credit.pkl", "rb") as f:
    x_treino, y_treino, X_teste, y_teste = pickle.load(f)

modelo_svm = SVC(kernel='rbf', C=105, random_state=1)
modelo_svm.fit(x_treino, y_treino)

# Previsões
previsoes = modelo_svm.predict(X_teste)

# Acurácia
acuracia = accuracy_score(y_teste, previsoes)

print(f"Melhor taxa de acerto manual (acurácia): {acuracia * 100:.2f}%\n\n\n")

# 4
# linear = 94.60%
# rbf = 98.20%
# poly = 96.80%
# sigmoid = 83.80%

#5
def discover_best_by_hand():
    C = 1.0
    acuracia_max = 0
    melhor_c = 1.0
    for i in range(0, 1000):
        modelo_svm = SVC(kernel='rbf', C=C, random_state=1)
        modelo_svm.fit(x_treino, y_treino)

        # Previsões
        previsoes = modelo_svm.predict(X_teste)

        # Acurácia
        acuracia = accuracy_score(y_teste, previsoes)
        if acuracia > acuracia_max:
            acuracia_max = acuracia
            melhor_c = C
        print(f"Taxa de acerto (acurácia): {acuracia * 100:.2f}%")
        C+= 0.1
    print(acuracia_max)
    print(C)
#discover_best_by_hand()

#6
# Definindo os hiperparâmetros para o GridSearch
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 50, 100, 105, 200],           # Inclui seu valor 105 para comparação
    'degree': [2, 3, 4],                            # só usado para 'poly'
    'gamma': ['scale', 'auto']                      # usado para 'rbf', 'poly', 'sigmoid'
}

# Instanciando o GridSearchCV
grid_search = GridSearchCV(SVC(random_state=1), param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(x_treino, y_treino)

print("\nMelhor combinação de hiperparâmetros encontrada pelo Grid Search:")
print(grid_search.best_params_)

melhor_modelo = grid_search.best_estimator_
previsoes_grid = melhor_modelo.predict(X_teste)
acuracia_grid = accuracy_score(y_teste, previsoes_grid)
print(f"Modelo com Grid Search - acurácia: {acuracia_grid*100:.2f}%")

print("\nRelatório de classificação do melhor modelo:")
print(classification_report(y_teste, previsoes_grid))



"""## Base de Dados Credit Data

# Análise dos resultados dos 4 algoritmos utilizados:

6. O resultado do SVM é melhor que os resultados do Naive Bayes, Florestas Aleatórias e Regressão Logística? Descreva sua análise de resultados (observe que para isso você deverá visualizar os resultados da Matriz de Confusão, acurácia, precisão e recall).
"""

""""
R: 

"""

