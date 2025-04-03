import scipy.io as scipy
import matplotlib.pyplot as plt
import numpy as np

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















