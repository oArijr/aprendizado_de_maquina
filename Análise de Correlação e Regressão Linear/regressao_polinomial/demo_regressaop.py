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
list_n1_train = regressao_polinomial(x, y, 1)
ax.plot(x, list_n1_train, c='r')

#D)
list_n2_train = regressao_polinomial(x, y, 2)
ax.plot(x, list_n2_train, c='g')

#E)
list_n3_train = regressao_polinomial(x, y, 3)
ax.plot(x, list_n3_train, c='black')

#F)
list_n8_train = regressao_polinomial(x, y, 8)
ax.plot(x, list_n8_train, c='y')

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

    return soma_residuos / y_tam

print("\nEQM 1")
print(round(erro_quadratico_medio(list_n1_train, y), 5))

print("EQM 2")
print(round(erro_quadratico_medio(list_n2_train, y), 5))

print("EQM 3")
print(round(erro_quadratico_medio(list_n3_train, y), 5))

print("EQM 8")
print(round(erro_quadratico_medio(list_n8_train, y), 5))
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

list_n1_train = regressao_polinomial(x_train, y_train, 1)
ax.plot(x_train, list_n1_train, c='r')

list_n2_train = regressao_polinomial(x_train, y_train, 2)
ax.plot(x_train, list_n2_train, c='g')

list_n3_train = regressao_polinomial(x_train, y_train, 3)
ax.plot(x_train, list_n3_train, c='black')

list_n8_train = regressao_polinomial(x_train, y_train, 8)
ax.plot(x_train, list_n8_train, c='y')

plt.show()

#J)
# Calculando EQM para dados de teste
print("\nEQM nos dados de TESTE:")
print("EQM 1:", round(erro_quadratico_medio(list_n1_train, y_test), 5))
print("EQM 2:", round(erro_quadratico_medio(list_n2_train, y_test), 5))
print("EQM 3:", round(erro_quadratico_medio(list_n3_train, y_test), 5))
print("EQM 8:", round(erro_quadratico_medio(list_n8_train, y_test), 5))

#K)
# Calculando R2 para os dados de treino
r2_train_n1 = r2_score(y_train, list_n1_train)
r2_train_n2 = r2_score(y_train, list_n2_train)
r2_train_n3 = r2_score(y_train, list_n3_train)
r2_train_n8 = r2_score(y_train, list_n8_train)

print("\nR2 nos dados de treino:")
print("R2 N=1:", round(r2_train_n1, 5))
print("R2 N=2:", round(r2_train_n2, 5))
print("R2 N=3:", round(r2_train_n3, 5))
print("R2 N=8:", round(r2_train_n8, 5))


# Calculando R2 para os dados de teste
list_n1_test = regressao_polinomial(x_test, y_test, 1)
list_n2_test = regressao_polinomial(x_test, y_test, 2)
list_n3_test = regressao_polinomial(x_test, y_test, 3)
list_n8_test = regressao_polinomial(x_test, y_test, 8)

r2_test_n1 = r2_score(y_test, list_n1_test)
r2_test_n2 = r2_score(y_test, list_n2_test)
r2_test_n3 = r2_score(y_test, list_n3_test)
r2_test_n8 = r2_score(y_test, list_n8_test)

print("\nR2 nos dados de teste:")
print("R2 N=1:", round(r2_test_n1, 5))
print("R2 N=2:", round(r2_test_n2, 5))
print("R2 N=3:", round(r2_test_n3, 5))
print("R2 N=8:", round(r2_test_n8, 5))


#I)
# O modelo N8 aparenta ser o melhor porém acreditamos que ele se encaixa no overfitting.
# Em alguns cenários de testes, o N8 acaba tendo performando muito mal, mostrando sua vulnerabilidade.
# Já o N3 por ser mais equilibrado acaba performando melhor com os dados de teste em média.