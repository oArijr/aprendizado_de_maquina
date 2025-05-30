# -*- coding: utf-8 -*-
"""TRABALHO 3: PARTE 1 - Algoritmo Naïve Bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Y-yMmG5njAv0vRM2_hcTdFDuneopATt

# PARTE 1: Algoritmo Naïve Bayes

Nesta primeira parte do Trabalho você irá aplicar o algoritmo de Naïve Bayes na base de dados de risco de crédito discutida em aula. Para isso você deve primeiramente importar as bibliotecas necessárias.
"""
def print_title(title):
  print(f"\n\n{'=' * 70}")
  print(f"{title}:")
  print(f"{'=' * 70}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# importe a base de dados de risco de crédito e nomeie com: dataset_risco_credito
dataset_risco_credito = pd.read_csv('dataset_risco_credito.csv', encoding='utf-8')
# imprima a base de dados para uma primeira avaliação dos dados
#print(dataset_risco_credito)

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

le_y = LabelEncoder()
y_risco_credito = le_y.fit_transform(y_risco_credito)

le_x = {}
colunas = dataset_risco_credito.columns[:-1] # ["historia", "divida", "garantias", "renda"]
X_risco_credito = np.array(X_risco_credito)
for i in range(X_risco_credito.shape[1]):
  le = LabelEncoder()
  X_risco_credito[:, i] = le.fit_transform(X_risco_credito[:, i])
  le_x[colunas[i]] = le


print_title("Dataset sem Label:")
df = pd.DataFrame(X_risco_credito, columns=colunas)
df['classe'] = [c for c in y_risco_credito]
print(df.to_string(index=False))


print_title("Dataset com Label:")
x_decoded = []
for i in range(len(y_risco_credito)):
  linha_decoded = []
  for j in range(len(X_risco_credito[i])):
    value_decoded = le_x[colunas[j]].inverse_transform([X_risco_credito[i][j]])[0]
    linha_decoded.append(value_decoded)
  x_decoded.append(linha_decoded)

df = pd.DataFrame(x_decoded, columns=colunas)
df['classe'] = [le_y.inverse_transform([c])[0] for c in y_risco_credito]
print(df)


sns.countplot(x=le_y.inverse_transform(y_risco_credito))
plt.title("Distribuição das Classes (Risco de Crédito)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()



# Supondo que 'X_risco_credito' seja um numpy array com dados codificados
df_encoded = pd.DataFrame(X_risco_credito, columns=colunas)

# Adiciona a variável alvo codificada como numérica também
df_encoded["classe"] = y_risco_credito  # <- aqui usa o y codificado, não o inverse_transform ainda

# Agora sim, faz a matriz de correlação completa
corr = df_encoded.corr()

# Exibe o heatmap com seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlação entre variáveis")
plt.show()


df_plot = pd.DataFrame(x_decoded, columns=colunas)
df_plot['classe'] = le_y.inverse_transform(y_risco_credito)

fig = px.scatter(df_plot, x='historia', y='renda', color='classe',
                 title='História do crédito x Renda',
                 labels={'historia': 'História', 'renda': 'Renda'})
fig.show()

fig2 = px.scatter(df_plot, x='garantias', y='renda', color='classe',
                 title='Garantias x Renda',
                 labels={'garantias': 'Garantias', 'renda': 'Renda'})
fig2.show()



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

# i) história boa, dívida alta, garantias nenhuma, renda > 35
entrada_i = ["boa", "alta", "nenhuma", "acima_35"]

# ii) história ruim, dívida alta, garantias adequada, renda < 15
entrada_ii = ["ruim", "alta", "adequada", "0_15"]

predict = naiveb_risco_credito.predict(X_risco_credito)

entrada_i_encoded = []
entrada_ii_encoded = []
for i in range(4):
  entrada_i_encoded.append(le_x[colunas[i]].transform([entrada_i[i]])[0])
  entrada_ii_encoded.append(le_x[colunas[i]].transform([entrada_ii[i]])[0])

# Previsão
pred_i = naiveb_risco_credito.predict([entrada_i_encoded])
pred_ii = naiveb_risco_credito.predict([entrada_ii_encoded])

print_title("Resultados das Previsões")
print(f"Entrada I: {entrada_i} => Classe prevista: {le_y.inverse_transform(pred_i)[0]} (esperado: baixo)")
print(f"Entrada II: {entrada_ii} => Classe prevista: {le_y.inverse_transform(pred_ii)[0]} (esperado: alto)")


probs = naiveb_risco_credito.predict_proba([entrada_i_encoded, entrada_ii_encoded])
df_probs = pd.DataFrame(probs, columns=le_y.classes_, index=["Entrada I", "Entrada II"])
print("\n", df_probs)

df_probs.plot(kind='bar', figsize=(8, 4))
plt.title("Probabilidades para cada classe")
plt.ylabel("Probabilidade")
plt.xticks(rotation=0)
plt.show()