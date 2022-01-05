
### Este programa tiene la finalidad de calcular el retorno y 
### la varianza de un portafolio de activos dada la rentabilidad 
### de cada uno de ellos, su matriz de covarianzas y los pesos de 
### cada uno de los activos en el portafolio.

### Por otra parte, también realiza cálculos para obtener el vector de 
### pesos que genera el portafolio de máximo retorno, de mínima varianza 
### y por último el máximo ratio de sharpe.

import numpy as np

n = 3

mu_ = np.array([0.15, 0.10, 0.12])
sigma_ = np.array([
             [0.025, 0.058, 0.010],
             [0.058, 0.040, -0.002],
             [0.010, -0.002, 0.028]
             ])


def portafiolio(mu, sigma, w):
    """
    Esta función recibe tres variables que son; retorno, matriz de covarianza
    y pesos, dando como resultado; el retorno del portafolio y su volatilidad.

    Parametros
    ----------
    mu: NumPy Array
        Vector de N x 1 que contiene los retornos de los activos en el portafolio
    sigma: NumPy Array
        Matriz de N x 1 que contiene las covarianzas entre los activos del portafolio
    w: NumPy Array
        Vector de N x 1 que contiene el vector de pesos.

    Resultado
    ----------
    mu_p: float
        Retorno promedio ponderado del portafolio.
    sigma_p: float
        Volatilidad del portafolio

    """
    mu_p = mu_ @ w.T
    mu_p = np.round(mu_p.item() * 100 , 2)
    sigma_p = ((w @ sigma_) @ w.T)**(1/2)
    sigma_p = np.round(sigma_p.item() * 100 , 2)
    print(f"El retorno del portafiolio es de {mu_p}%\n",
                                f"Y su volatilidad {sigma_p}%") 
    return mu_p, sigma_p


### Resolución de sistema de ecuaciones para calcular los pesos 
### del portafolio de máximo retorno
a = np.array([
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]
             ])
x = np.linalg.solve(a,mu_)
ub = np.argmax(x)
w_maxr = a[ub] #Vector pesos portafolio máximo retorno

portafiolio(mu_,sigma_,w_maxr)

w1 = (np.linspace(0,1,21)) #Creación de grilla de búsqueda
w2 = (np.linspace(0,1,21))

pesos = []
for i in w1: 
    for j in w2:
        if i + j <=1:
            pesos.append([i,j])

sigma_aux = 100

for i in pesos: # Búsqueda de vector pesos para minima varianza
    x3 = (1-i[0]-i[1])
    wi = [
         i[0], i[1], x3
         ]
    wi = np.array(wi)
    sigma_x = ((wi @ sigma_) @ wi.T)
    if sigma_x < sigma_aux:
        sigma_aux = sigma_x
        w_min = [
                i[0], i[1], x3
                ]
        w_min = np.array(w_min) #Vector pesos portafolio minima varianza 

portafiolio(mu_,sigma_,w_min)

sharpe_aux = 0

for i in pesos: # Búsqueda de vector pesos para el máximo Ratio de Sharpe
    x3 = (1-i[0]-i[1])
    wi = [
         i[0], i[1], x3
         ]
    wi = np.array(wi)
    tetha_x = ((wi @ sigma_) @ wi.T)**(1/2)
    mu_x = mu_ @ wi.T
    sharpe = (mu_x - 0.02)/tetha_x
    if sharpe > sharpe_aux:
        sharpe_aux = sharpe
        w_max = [
                i[0], i[1], x3
                ]
        w_max = np.array(w_max) # Vector pesos portafolio máximo Ratio de Sharpe

portafiolio(mu_,sigma_,w_max)



