import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import pickle

"""##Algoritmo de Regressão Logística para uma base de dados maior (Credit Data)

7. Agora aplique a Regressão Logística na base de dados ‘credit.pkl’. De quanto foi a taxa de acerto?

8. O resultado com a base de dados ‘credit.pkl’ é melhor que os resultados do Naive Bayes e das Florestas Aleatórias? Descreva sua análise de resultados (observe que para isso você deverá visualizar os resultados da Matriz de Confusão, acurácia, precisão e recall).
"""
with open("credit.pkl", "rb") as f:
    X_risco_credito, y_risco_credito, X_teste, y_teste = pickle.load(f)
# print(X_teste[:20])
# print(y_teste[:20])
# print(X_risco_credito[:20])
# print(y_risco_credito[:20])


# Regressao
modelo = LogisticRegression(random_state=1)
modelo.fit(X_risco_credito, y_risco_credito) # Treino

# Previsões
previsoes = modelo.predict(X_teste)

# Acurácia
acuracia = accuracy_score(y_teste, previsoes)
print(f"Taxa de acerto (acurácia): {acuracia * 100:.2f}%")

# Precisão
precisao = precision_score(y_teste, previsoes, average='macro')
print(f"\nPrecisão: {precisao * 100:.2f}%")

# Recall
recall = recall_score(y_teste, previsoes, average='macro')
print(f"\nRecall: {recall * 100:.2f}%")


# Matriz de confusão
matriz = confusion_matrix(y_teste, previsoes)
df_matriz = pd.DataFrame(
    matriz,
    index=["Real 0", "Real 1"],
    columns=["Previsto 0", "Previsto 1"]
)
print("\nMatriz de Confusão:")
print(df_matriz.to_string())

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes))