import numpy as np

n = 3
mu_ = np.array([0.15, 0.10, 0.12])
sigma_ = np.array([[0.025, 0.058, 0.010],[0.058, 0.040, -0.002],[0.010, -0.002, 0.028]])


def random_w(n):
    '''
    Función que crea unos pesos aleatorios para el portafolio 
    '''
    k = np.random.rand(n)
    return k / sum(k)

w_ = np.array([0.25, 0.25, 0.50])

mu_p = mu_ @ w_.T
mu_p = np.round(mu_p.item() * 100 , 2)
sigma_p = ((w_ @ sigma_) @ w_.T)**(1/2)
sigma_p = np.round(sigma_p.item() * 100 , 2)

# print(np.linalg.solve(mu_, mu_p))

def max_r():
    '''
    Función que retorna el w de máximo retorno dado que no se toma en cuenta el riesgo

    '''
    a = np.array([[1,0,0],[0,1,0],[0,0,1]])
    x = np.linalg.solve(a,mu_)
    ub = np.argmax(x)
    return a[ub]

print(max_r())


#MINIMO CULIAO

w1 = (np.linspace(0,1,21))
w2 = (np.linspace(0,1,21))

pesos = []
for i in w1:
    for j in w2:
        if i + j <=1:
            pesos.append([i,j])

sigma_aux = 100

for i in pesos:
    x3 = (1-i[0]-i[1])
    wi = [i[0],i[1],x3]
    wi = np.array(wi)
    sigma_x = ((wi @ sigma_) @ wi.T)
    if sigma_x < sigma_aux:
        sigma_aux = sigma_x
        w_aux = [i[0],i[1],x3]
print(w_aux)




    