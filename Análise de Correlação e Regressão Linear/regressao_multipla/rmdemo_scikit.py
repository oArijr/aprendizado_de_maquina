import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression

# i)

mat = sio.loadmat('data.mat')
data = mat['data']

tamanho = [casa[0] for casa in data]
quartos = [casa[1] for casa in data]
precos = [casa[2] for casa in data]

x = np.column_stack((tamanho, quartos)) # Empilha arrays unidimensionais como colunas de uma matriz 2D,
modelo = LinearRegression().fit(x, precos)
b0, b1, b2 = modelo.intercept_, modelo.coef_[0], modelo.coef_[1]

preco_1650_3 = b0 + b1 * 1650 + b2 * 3
print(f"Preço previsto para casa com 1650m² e 3 quartos: {preco_1650_3:.2f}")
print(f"Equação: Preço = {b0:.2f} + {b1:.2f}×(Tamanho) + {b2:.2f}×(Quartos)")