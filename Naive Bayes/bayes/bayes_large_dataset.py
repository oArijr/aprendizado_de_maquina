import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score



with open("credit.pkl", "rb") as f:
  X_risco_credito, y_risco_credito, X_teste, y_teste = pickle.load(f)
# print(X_teste[:20])
# print(y_teste[:20])
# print(X_risco_credito[:20])
# print(y_risco_credito[:20])


from sklearn.naive_bayes import GaussianNB
modelo = GaussianNB()
modelo.fit(X_risco_credito, y_risco_credito) # Treino

# Previsões
previsoes = modelo.predict(X_teste)

# Acurácia
acuracia = accuracy_score(y_teste, previsoes)
print(f"Taxa de acerto (acurácia): {acuracia * 100:.2f}%")

# Matriz de confusão
matriz = confusion_matrix(y_teste, previsoes)
df_matriz = pd.DataFrame(
  matriz,
  index=["Real 0", "Real 1"],
  columns=["Previsto 0", "Previsto 1"]
)
print("\nMatriz de Confusão:")
print(df_matriz.to_string())

# Precisão
precisao = precision_score(y_teste, previsoes, average='macro')
print(f"\nPrecisão: {precisao * 100:.2f}%")

# Recall
recall = recall_score(y_teste, previsoes, average='macro')
print(f"\nRecall: {recall * 100:.2f}%")