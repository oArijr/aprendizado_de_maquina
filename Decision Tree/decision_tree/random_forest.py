
"""
# Algoritmo Random Forest

Nesta seção iremos utilizar o algoritmo Random Forest para a mesma base de crédito (**Credit Risk Dataset**) - arquivo *credit.pkl*.

a) Importe o pacote RandomForestClassifier do sklearn para treinar o seu algoritmo de floresta randomica.
"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

with open('credit.pkl', 'rb') as f:
  X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

from sklearn.ensemble import RandomForestClassifier

"""
b) Para gerar a classificação você deve adicionar alguns parâmetros:
*   n_estimators=10  --> número de árvores que você irá criar
*   criterion='entropy'
*   random_state = 0
"""
floresta_credit = RandomForestClassifier(
    n_estimators=10,
    criterion='entropy',
    random_state=0
)

floresta_credit.fit(X_credit_treinamento, y_credit_treinamento)

"""
c) Faça a previsão com os dados de teste. Visualize os dados e verifique se as previsões estão de acordo com os dados de teste (respostas reais).
"""
floresta_predict = floresta_credit.predict(X_credit_teste)

resultado_fp = pd.DataFrame({
    'Real': y_credit_teste,
    'Previsto': floresta_predict
})

print(resultado_fp.head(20))

"""
d) Agora faça o cálculo da acurácia para calcular a taxa de acerto entre os valores reais (y teste) e as previsões. O resultado foi melhor do que a árvore de decisão simples?
"""
floresta_accuracy = accuracy_score(y_credit_teste, floresta_predict) * 100
print(f"Acurácia do modelo Random Forest: {floresta_accuracy}%")

# Não, o resultado da árvore simples foi de 98.2% já utilizando a floresta a acuracia chegou a 96.8%.

"""
e) Se o resultado foi inferior, como você poderia resolver isso? Quais foram os resultados obtidos?
"""
floresta_credit_novo = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    random_state=0
)

floresta_credit_novo.fit(X_credit_treinamento, y_credit_treinamento)

floresta_predict_novo = floresta_credit_novo.predict(X_credit_teste)

floresta_accuracy_novo = accuracy_score(y_credit_teste, floresta_predict_novo) * 100
print(f"Nova Acurácia do modelo Random Forest: {floresta_accuracy_novo}%")

# Aumentei a quantidade de árvores para 100 e o resultado igualou a acuracia entre a floresta e uma única árvore, ambas com 98.2% de acurácia.
"""
f) Faça a análise da Matriz de Confusão.
"""
matriz = confusion_matrix(y_credit_teste, floresta_predict_novo)
df_matriz = pd.DataFrame(
    matriz,
    index=["Real 0", "Real 1"],
    columns=["Previsto 0", "Previsto 1"]
)
print("\nMatriz de Confusão:")
print(df_matriz.to_string(), "\n")

# Foram classificados 433 corretamente como risco alto
# Foram classificados 3 incorretamente como risco baixo
# Foram classificados 58 corretamente como risco baixo.
# Foram classificados 6 incorretamente como risco alto.

# O modelo apresentou uma excelente taxa de acerto. Porem o modelo apresentou uma leve tendência a classificar os clientes com um risco alto.

"""
g) Faça um print do parâmetro classification_report entre os dados de teste e as previsões. Explique qual é a relação entre precision e recall nos dados. Como você interpreta esses dados?
"""
print(classification_report(y_credit_teste, floresta_predict_novo))

# Precision indica a proporção de previsões positivas que estavam corretas. No caso da classe 0 (risco alto) a precisão foi de 0.99,
# o que quer dizer que 99% dos clientes previstos como classe 0 (risco alto) realmente são classe 0 (risco alto) e para a classe 1 (risco baixo),
# a precisão foi de 0.91, ou seja, 91% dos clientes previstos como classe 1 (risco baixo) realmente são classe 1 (risco  baixo).

# Recall mostra a proporção de casos positivos reais que foram corretamente identificados pelo modelo.
# Para a classe 0 (risco alto), o recall foi de 0.99, indicando que 99% dos classes 0 foram corretamente detectados.
# Para a classe 1 (risco baixo), o recall foi de 0.95, o que significa que 95% dos classes 1 foram corretamente detectados.

# Como falei na resposta anterior, o modelo apresentou uma leve tendencia a classificar como risco alto.