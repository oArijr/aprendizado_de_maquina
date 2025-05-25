import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def print_title(title):
    print(f"\n\n{'=' * 60}")
    print(f"{title}:")
    print(f"{'=' * 60}")

"""# PARTE 3: Regressão Logística

1. Utilize a base de dados construída no Trabalho 3 ‘risco_credito.pkl’, que possui 14 registros, para testar o algoritmo de Regressão Logística.

2. Faça o Encoder dos dados e, para facilitar, como fizemos na aula teórica, apague os registros que possuem a classe ‘moderado’. No total teremos 11 registros.

3. Treine o algoritmo de regressão logística e utilize o parâmetro ‘random_state =1’ para ter sempre o mesmo resultado.

4. Utilize o comando ‘.intercept_’ para ter o resultado do B0.
O resultado deve ser =-0.80828993

5. Utilize o comando ‘.coef_’ para ter o resultado dos demais parâmetros que deve ser:
array([[-0.76704533,  0.23906678, -0.47976059,  1.12186218]])

6. Agora utilize o comando ‘predict’ para fazer o teste do seu algoritmo com:

    a) história boa, dívida alta, garantias nenhuma, renda > 35
    (o resultado desse teste deve ser ‘baixo’)

    b) história ruim, dívida alta, garantias adequada, renda < 15
    (o resultado desse teste deve ser ‘alto’)
"""
with open("risco_credito.pkl", "rb") as f:
    X_risco_credito, y_risco_credito = pickle.load(f)


# LabelEncoder
le_y = LabelEncoder()
y_risco_credito = le_y.fit_transform(y_risco_credito)

le_x = {}
colunas = ["historia", "divida", "garantias", "renda"]
X_risco_credito = np.array(X_risco_credito)
for i in range(X_risco_credito.shape[1]):
    le = LabelEncoder()
    X_risco_credito[:, i] = le.fit_transform(X_risco_credito[:, i])
    le_x[colunas[i]] = le


# Remove o Moderado (classe 2)
x_filtered = []
y_filtered = []
for i in range(len(y_risco_credito)):
    if le_y.inverse_transform([y_risco_credito[i]])[0] != "moderado":
        x_filtered.append(X_risco_credito[i])
        y_filtered.append(y_risco_credito[i])



print_title("Dataset sem Label:")
df = pd.DataFrame(x_filtered, columns=colunas)
df['classe'] = [c for c in y_filtered]
print(df.to_string(index=False))


print_title("Dataset com Label:")
x_decoded = []
for i in range(len(x_filtered)):
    linha_decoded = []
    for j in range(len(x_filtered[i])):
        value_decoded = le_x[colunas[j]].inverse_transform([x_filtered[i][j]])[0]
        linha_decoded.append(value_decoded)
    x_decoded.append(linha_decoded)

df = pd.DataFrame(x_decoded, columns=colunas)
df['classe'] = [le_y.inverse_transform([c])[0] for c in y_filtered]
print(df)


from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(random_state=1)
modelo.fit(x_filtered, y_filtered) # Treino

b0 = modelo.intercept_
outros_b = modelo.coef_

print_title("Betas")
print("B0 =", b0[0])
for i, coef in enumerate(outros_b[0]):
    print(f"B{i+1} =", coef)



# i) história boa, dívida alta, garantias nenhuma, renda > 35
entrada_i = ["boa", "alta", "nenhuma", "acima_35"]

# ii) história ruim, dívida alta, garantias adequada, renda < 15
entrada_ii = ["ruim", "alta", "adequada", "0_15"]

entrada_i_encoded = []
entrada_ii_encoded = []
for i in range(4):
    entrada_i_encoded.append(le_x[colunas[i]].transform([entrada_i[i]])[0])
    entrada_ii_encoded.append(le_x[colunas[i]].transform([entrada_ii[i]])[0])

# Previsão
pred_i = modelo.predict([entrada_i_encoded])
pred_ii = modelo.predict([entrada_ii_encoded])

print_title("Resultados das Previsões")
print(f"Entrada I: {entrada_i} => Classe prevista: {le_y.inverse_transform(pred_i)[0]} (esperado: baixo)")
print(f"Entrada II: {entrada_ii} => Classe prevista: {le_y.inverse_transform(pred_ii)[0]} (esperado: alto)")