import scipy.io as scipy
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import r2_score

# Ari Elias da Silva Júnior e Luigi Garcia Marchetti
def regressao_polinomial(x, y, n):
    lista_beta = np.flip(np.polyfit(x, y, n))
    x = np.array(x)
    r = lista_beta[0] + lista_beta[1] * x
    for i in range(2, len(lista_beta)):
        r += lista_beta[i] * (x ** i)
    return r

# A)
mat = scipy.loadmat('data_preg.mat')
data = mat['data']

x = [coluna[0] for coluna in data]
y = [coluna[1] for coluna in data]

# B)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
ax.set_title("Normal")
ax.scatter(x, y)

#C)
list_n1 = regressao_polinomial(x, y, 1)
ax.plot(x , list_n1, c='r')

#D)
list_n2 = regressao_polinomial(x, y, 2)
ax.plot(x , list_n2, c='g')

#E)
list_n3 = regressao_polinomial(x, y, 3)
ax.plot(x , list_n3, c='black')

#F)
list_n8 = regressao_polinomial(x, y, 8)
ax.plot(x, list_n8, c='y')

plt.show()

#G)
media_y = np.mean(y)

def residuo(y, media_y):
    return np.pow(y - media_y, 2)

def erro_quadratico_medio(y_linear, y):
    soma_residuos = 0
    for i, j in zip(y_linear, y):
        soma_residuos += residuo(i, j)

    y_tam = len(y_linear)

    res = soma_residuos / y_tam
    return res

print("\nEQM 1")
print(erro_quadratico_medio(list_n1, y))

print("EQM 2")
print(erro_quadratico_medio(list_n2, y))

print("EQM 3")
print(erro_quadratico_medio(list_n3, y))

print("EQM 8")
print(erro_quadratico_medio(list_n8, y))
print()

#H)
x_y = list(zip(x, y))
test_set = random.sample(x_y, int(len(x_y) * 0.1))
print("Test Set:")
for pair in test_set:
    print(f"({pair[0]}, {pair[1]})")

treino_set = [item for item in x_y if item not in test_set]
x_train, y_train = zip(*treino_set)
x_test, y_test = zip(*test_set)

#I)
fig2 = plt.figure(figsize=(10,8))
ax = fig2.add_subplot()
ax.set_title("\nTreino")
ax.scatter(x_train, y_train)

list_n1 = regressao_polinomial(x_train, y_train, 1)
ax.plot(x_train , list_n1, c='r')

list_n2 = regressao_polinomial(x_train, y_train, 2)
ax.plot(x_train , list_n2, c='g')

list_n3 = regressao_polinomial(x_train, y_train, 3)
ax.plot(x_train , list_n3, c='black')

list_n8 = regressao_polinomial(x_train, y_train, 8)
ax.plot(x_train, list_n8, c='y')

plt.show()

#J)
# Obter os coeficientes dos modelos treinados
coef_n1_test = regressao_polinomial(x_test, y_test, 1)
ax.plot(x_train , list_n1, c='r')

coef_n2_test = regressao_polinomial(x_test, y_test, 2)
ax.plot(x_train , list_n2, c='g')

coef_n3_test = regressao_polinomial(x_test, y_test, 3)
ax.plot(x_train , list_n3, c='black')

coef_n8_test = regressao_polinomial(x_test, y_test, 8)
ax.plot(x_train, list_n8, c='y')

# Calculando EQM para dados de teste
print("\nEQM nos dados de TESTE:")
print("EQM 1:", erro_quadratico_medio(coef_n1_test, y_test))
print("EQM 2:", erro_quadratico_medio(coef_n2_test, y_test))
print("EQM 3:", erro_quadratico_medio(coef_n3_test, y_test))
print("EQM 8:", erro_quadratico_medio(coef_n8_test, y_test))

#K)
# Calculando R2 para os dados de treino
r2_train_n1 = r2_score(y_train, list_n1)
r2_train_n2 = r2_score(y_train, list_n2)
r2_train_n3 = r2_score(y_train, list_n3)
r2_train_n8 = r2_score(y_train, list_n8)

print("\nR2 nos dados de TREINO:")
print("R2 N=1:", r2_train_n1)
print("R2 N=2:", r2_train_n2)
print("R2 N=3:", r2_train_n3)
print("R2 N=8:", r2_train_n8)

# Calculando R2 para os dados de teste
r2_test_n1 = r2_score(y_test, coef_n1_test)
r2_test_n2 = r2_score(y_test, coef_n2_test)
r2_test_n3 = r2_score(y_test, coef_n3_test)
r2_test_n8 = r2_score(y_test, coef_n8_test)

print("\nR2 nos dados de TESTE:")
print("R2 N=1:", r2_test_n1)
print("R2 N=2:", r2_test_n2)
print("R2 N=3:", r2_test_n3)
print("R2 N=8:", r2_test_n8)


# O modelo N8 em geral. Em alguns cenários de testes, devido principalmente a serem poucos casos, isso pode variar,
# com o N3 por exemplo desempenhando melhor