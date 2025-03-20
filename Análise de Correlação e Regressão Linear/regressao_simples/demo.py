import matplotlib.pyplot as plt
import numpy as np
import regressao

x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y2 = np.array([9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

x3 = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
y3 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])

data_list = [
    {"x": x1, "y": y1, "title": "Dataset 1"},
    {"x": x2, "y": y2, "title": "Dataset 2"},
    {"x": x3, "y": y3, "title": "Dataset 3"}
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, data in zip(axes, data_list):
    x, y = data["x"], data["y"]

    correlacao = regressao.correlacao(x, y)
    regressao_linear = regressao.regressao(x, y)
    b0 = regressao.calcular_b0(x, y)
    b1 = regressao.calcula_b1(x, y)

    ax.scatter(x, y)
    ax.plot(x, regressao_linear, label="Linha de Regressão", color='red', linewidth=2)


    ax.set_title(f"{data['title']}\nCorrelação: {correlacao:.3f}\nB0: {b0:.3f}\nB1: {b1:.3f}", fontweight='bold')
    ax.set_xlabel("X - Variáveis Independentes", fontsize=12)
    ax.set_ylabel("Y - Variáveis Dependentes", fontsize=12)

plt.tight_layout()
plt.show()

# Qual dos datasets não é apropriado para regressão linear?
#
# R:
# O terceiro dataset não é apropriado para regressão linear.
# A variável x3 tem um valor constante (8) para a maioria dos pontos, logo não existe variabilidade
# para se traçar uma linha de regressão. Ainda por cima existe um outlier também dificultando a regressão.