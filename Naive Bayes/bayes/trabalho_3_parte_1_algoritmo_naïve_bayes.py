# -*- coding: utf-8 -*-
"""TRABALHO 3: PARTE 1 - Algoritmo Naïve Bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Y-yMmG5njAv0vRM2_hcTdFDuneopATt

# PARTE 1: Algoritmo Naïve Bayes

Nesta primeira parte do Trabalho você irá aplicar o algoritmo de Naïve Bayes na base de dados de risco de crédito discutida em aula. Para isso você deve primeiramente importar as bibliotecas necessárias.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def set_label(classe):
  if classe == 0:
    return "Alto"
  elif classe == 1:
    return "Baixo"
  elif classe == 2:
    return "Moderado"
  else:
    raise Exception("Classe invalida")

# importe a base de dados de risco de crédito e nomeie com: dataset_risco_credito
dataset_risco_credito = pd.read_csv('dataset_risco_credito.csv', encoding='utf-8')
# imprima a base de dados para uma primeira avaliação dos dados
print(dataset_risco_credito)

"""# 1 - Pré-Processamento dos dados

a) DIVISÃO DA BASE DE DADOS

Separe a base de dados dataset_risco_credito em:
 - variável x, com nome: X_risco_credito
 - classe y, com nome: y_risco_credito

DICA: você pode utilizar .iloc para selecionar as colunas da matriz e .values para converter para o numpy array.

b) APLICAR LABEL ENCODER

Perceba que seus dados possuem atributos categóricos (string). Porém, para aplicar esses dados em um algoritmo de aprendizado você precisa transformá-lo em atributo numérico. 

Como você pode resolver isso?

DICA: Veja o que é e como aplicar o Label Enconder em: https://youtu.be/nLKEkBAbpQo
"""

X_risco_credito = dataset_risco_credito.iloc[:, :-1].values
y_risco_credito = dataset_risco_credito.iloc[:, -1].values

"""c) SALVAR O ARQUIVO PRÉ-PROCESSADO"""
# Salvar o arquivo:
import pickle
with open('risco_credito.pkl', 'wb') as f:
  pickle.dump([X_risco_credito, y_risco_credito], f)


# Resultado do LabelEncoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
for i in range(X_risco_credito.shape[1]):
  X_risco_credito[:, i] = labelencoder.fit_transform(X_risco_credito[:, i])
print("\n\nX_risco_credito:\n", X_risco_credito)

y_risco_credito = labelencoder.fit_transform(y_risco_credito)
print("\n\nY_risco_credito:\n", y_risco_credito)



"""# 2 - Algoritmo Naïve Bayes"""
# importar da biblioteca sklearn o pacote Nayve Bayes
# utilizamos a distribuição estatística Gaussiana (classe GaussianNB) ou distribuição normal pois é mais usado para problemas genéricos
from sklearn.naive_bayes import GaussianNB

# Criar o objeto Nayve Bayes
naiveb_risco_credito = GaussianNB()

"""a) TREINAR O ALGORITMO

Para treinar o algoritmo, você deve gerar a tabela de probabilidades. Para isso, você pode utilizar **.fit** para gerar a tabela.

DICA: O 1º parametro são os atributos/características (x) e o 2º parametro é a classe (y).

OBS: Não se preocupe, o algoritmo faz a correção laplaciana automaticamente :) .

b) FAZER A PREVISÃO

Utilize **.predict** para fazer a previsão realizada no exemplo em sala.

i) história boa, dívida alta, garantia nenhuma, renda > 35

ii) história ruim, dívida alta, garantia adequada, renda < 15

Verifique nos slides se seu resultado está correto!
"""
naiveb_risco_credito.fit(X_risco_credito, y_risco_credito)

predict = naiveb_risco_credito.predict(X_risco_credito)

print()
df_resultado = pd.DataFrame(X_risco_credito, columns=dataset_risco_credito.columns[:-1])
df_resultado['Classe Real'] = [set_label(c) for c in y_risco_credito]
df_resultado['Previsão'] = [set_label(p) for p in predict]
print(df_resultado)


# utilize .class_count_ para contar quantos registros tem em cada classe
class_count = naiveb_risco_credito.class_count_
print("\n\nClass count REAL:")
for i in range(len(class_count)):
  print(set_label(i), "=", class_count[i])



from sklearn.metrics import accuracy_score
acuracia = accuracy_score(y_risco_credito, predict)

# Cálculo da margem de erro
margem_erro = 1 - acuracia

print(f"\nAcurácia do modelo: {acuracia * 100:.2f}%")
print(f"Margem de erro do modelo: {margem_erro * 100:.2f}%")