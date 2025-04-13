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

    return soma_residuos / y_tam

print("\nEQM 1")
print(round(erro_quadratico_medio(list_n1, y), 5))

print("EQM 2")
print(round(erro_quadratico_medio(list_n2, y), 5))

print("EQM 3")
print(round(erro_quadratico_medio(list_n3, y), 5))

print("EQM 8")
print(round(erro_quadratico_medio(list_n8, y), 5))
print()

#H)
x_y = list(zip(x, y))
test_set = random.sample(x_y, int(len(x_y) * 0.1))
print(test_set)

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
coef_n1 = np.polyfit(x_train, y_train, 1)
coef_n2 = np.polyfit(x_train, y_train, 2)
coef_n3 = np.polyfit(x_train, y_train, 3)
coef_n8 = np.polyfit(x_train, y_train, 8)

# Calcular predições nos dados de teste
test_n1 = np.polyval(coef_n1, x_test)
test_n2 = np.polyval(coef_n2, x_test)
test_n3 = np.polyval(coef_n3, x_test)
test_n8 = np.polyval(coef_n8, x_test)

# Calculando EQM para dados de teste
print("\nEQM nos dados de TESTE:")
print("EQM 1:", round(erro_quadratico_medio(test_n1, y_test), 5))
print("EQM 2:", round(erro_quadratico_medio(test_n2, y_test), 5))
print("EQM 3:", round(erro_quadratico_medio(test_n3, y_test), 5))
print("EQM 8:", round(erro_quadratico_medio(test_n8, y_test), 5))

#K)
# Calculando R2 para os dados de treino
r2_train_n1 = r2_score(y_train, list_n1)
r2_train_n2 = r2_score(y_train, list_n2)
r2_train_n3 = r2_score(y_train, list_n3)
r2_train_n8 = r2_score(y_train, list_n8)

print("\nR2 nos dados de TREINO:")
print("R2 N=1:", round(r2_train_n1, 5))
print("R2 N=2:", round(r2_train_n2, 5))
print("R2 N=3:", round(r2_train_n3, 5))
print("R2 N=8:", round(r2_train_n8, 5))

# Calculando R2 para os dados de teste
r2_test_n1 = r2_score(y_test, test_n1)
r2_test_n2 = r2_score(y_test, test_n2)
r2_test_n3 = r2_score(y_test, test_n3)
r2_test_n8 = r2_score(y_test, test_n8)

print("\nR2 nos dados de TESTE:")
print("R2 N=1:", round(r2_test_n1, 5))
print("R2 N=2:", round(r2_test_n2, 5))
print("R2 N=3:", round(r2_test_n3, 5))
print("R2 N=8:", round(r2_test_n8, 5))




degrees = [1, 2, 3, 8]
r2_train = [r2_train_n1, r2_train_n2, r2_train_n3, r2_train_n8]
r2_test = [r2_test_n1, r2_test_n2, r2_test_n3, r2_test_n8]

fig4 = plt.figure(figsize=(10, 6))
ax = fig4.add_subplot()
ax.plot(degrees, r2_train, 'o-', label='R2 Treino')
ax.plot(degrees, r2_test, 's-', label='R2 Teste')
ax.set_xlabel('Grau do Polinômio (N)')
ax.set_ylabel('R2')
ax.set_title('Comparação de R2 entre dados de treino e teste')
ax.legend()
plt.grid(True)
plt.show()