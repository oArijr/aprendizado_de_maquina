import scipy.io as scipy
import matplotlib.pyplot as plt
import numpy as np
# A)
mat = scipy.loadmat('data_preg.mat')
data = mat['data']

x = [coluna[0] for coluna in data]
y = [coluna[1] for coluna in data]
n = 2

# B)
plt.figure(figsize=(10,8))
plt.scatter(x, y)

plt.show()

#C)
result = np.polyfit(x, y, n)
result_flip = np.flip(result)
print(result)
print(result_flip)

#D)

def regressao_polinomial(x, y, n):






