import math
import numpy as np
def distancia_euclidiana(x1, x2, y1, y2):
    return math.sqrt(sum(np.pow(x2 - x1, 2) , np.pow(y2 - y1, 2)))

def classificador():
