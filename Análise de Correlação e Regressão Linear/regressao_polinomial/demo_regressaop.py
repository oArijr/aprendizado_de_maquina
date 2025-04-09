import scipy.io as scipy
import matplotlib.pyplot as plt
import numpy as np
import random

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
list_n3 = regressao_polinomial(x, y, 8)
ax.plot(x , list_n3, c='y')

plt.show()

#G)
media_y = np.mean(y)

def residuo(y, media_y):
    return np.pow(y - media_y, 2)

soma_residuos = 0
for i in y:
    soma_residuos += residuo(i, media_y)

y_tam = len(y)

eqm = soma_residuos / y_tam
print(eqm)

#H)
x_y = list(zip(x, y))
test_set = random.sample(x_y, int(len(x_y) * 0.1))
print(test_set)

treino_set = [item for item in x_y if item not in test_set]
x_train, y_train = zip(*treino_set)
x_test, y_test = zip(*test_set)




