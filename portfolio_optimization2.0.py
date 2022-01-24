import pandas as pd 
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from pandas_datareader import DataReader as wb
import seaborn as sns
from scipy.optimize import minimize 

tickers = ['TSLA', 'AAPL', 'FB', 'NVDA', 'MSFT', 'AMZN'] #tickers de activos
periodo = (dt.datetime.now() - relativedelta(years=5))   #periodo de simulación
df = yf.download(tickers,periodo)['Adj Close']           #creación de df con api de yf  

ret = (df/df.shift(1) -1).dropna()            #retornos discretos
ret_log = np.log(df/df.shift(1)).dropna()     #retornos continuos 

w = np.random.rand(len(tickers))              # w aleatorios
w /= np.sum(w)   

#Métricas 
ret_mean = ret_log.mean().values             #retorno promedio de cada activo
port_mean = np.dot(w.T,ret_mean)             #retorno promedio del portafolio ajustado por el peso
cov = ret_log.cov()                          #matriz varianza covarianza 
desv = np.sqrt(np.dot(w.T,np.dot(cov,w)))    #desviación estándar diaria del portafolio
lp = np.dot(df.iloc[-1].values,w.T)          #último valor del portafolio ajustado por pesos


#Funciones
def metrics(weights=[]):
    mean = ret_log.mean().values    #vector retornos esperados 
    cov = ret_log.cov()             
    w = np.array(weights)
    ret = np.dot(mean.T,w) * 252    #retorno del portafolio anual
    desv = np.sqrt(np.dot(w.T,np.dot(cov * 252,w))) #volatilidad anual del portafolio 
    return np.array([ret,desv,ret/desv])

def max_ret(weights=[]):
    return -metrics(weights)[0]

def min_desv(weights=[]):
    return metrics(weights)[1]

def max_sharpe(weights=[]):
    return -metrics(weights)[2]

#Optimizacion
conds = {'type': 'eq', 'fun' :lambda x: np.sum(x) -1} #condiciones para la optimización 
limits = tuple((0,1) for x in range(len(tickers)))    #limites para la optimización 
weights = len(tickers) * [1./len(tickers)]            #vector pesos inicial

opti = minimize(max_ret, weights, method='SLSQP', bounds=limits,constraints=conds) #para obtener máximo retorno, minínima volatilidad o máximo sharpe, cambiar la primera variable por una de las funciones anteirores
w_sharpe = opti.x

#Resultados
op_w = list(np.round(opti.x,3))
op_ret = np.round(metrics(op_w)[0],3)
op_desv = np.round(metrics(op_w)[1],3)
op_sharpe = np.round(metrics(op_w)[2],3)

resultados = '\nResultados optimización\n'\
+ 'Activos: ' + str(tickers) + '\n'\
+ 'Pesos: ' + str(op_w) + '\n'\
+ '\nRetorno anual : ' + str(op_ret) + '\n'\
+ 'Volatilidad anual: ' + str(op_desv) + '\n'\
+ 'Ratio de sharpe: ' + str(op_sharpe)

print(resultados)

#Test montecarlo
def montecarlo(data,tickers,simul):
    monte_ret = []
    monte_desv = []
    monte_sharpe = []
    weights = []
    ret = data.mean().values
    cov = data.cov()
    df1 = pd.DataFrame(columns=tickers)
    for i in range(simul):
        w = np.random.rand(len(tickers))
        w = w/w.sum()
        d1 = np.dot(w.T,ret)*252
        d2 = np.sqrt(np.dot(w.T,np.dot(cov,w))*252)
        d3 = d1/d2
        monte_ret.append(d1)
        monte_desv.append(d2)
        monte_sharpe.append(d3)
        weights.append(w)
    df1 = pd.DataFrame(columns=tickers, data=weights)
    df2 = pd.DataFrame(columns=['Ret', 'Desv', 'Sharpe'],)
    df2['Ret'] = monte_ret
    df2['Desv'] = monte_desv
    df2['Sharpe'] = monte_sharpe
    df = pd.concat([df1,df2],axis=1)
    return df

def plot_portfolios(port,sharpe,tickers):
    opt_port = op_w
    x = port['Desv'].values
    y= port['Ret'].values
    x_op = op_desv
    y_op = op_ret
    plt.figure()
    plt.scatter(x, y,c=y/x,cmap='RdYlGn')
    plt.title('Optimización de portafolio | ' + str(T) + ' simulaciones')
    plt.ylabel('|Rentabilidad anual|')
    plt.xlabel(
    '|Volatilidad anual|'\
        + '\n'+ '---------------------------'\
        + '\n' + 'Activos: ' + str(tickers)\
        + '\n' + 'Pesos: ' + str(opt_port)
        + "\n" + "Rentabilidad del portafolio: " + str(op_ret)
        + "\n" + "Volatilidad del portafolio: " + str(op_desv))
    sns.set_theme()
    plt.colorbar()
    plt.scatter(x_op, y_op,color='purple')
    plt.show()

T = 1000000
port_montecarlo = montecarlo(ret_log,tickers,T)
max_sharpe = port_montecarlo[port_montecarlo['Sharpe'] == port_montecarlo['Sharpe'].max()]
fig = plot_portfolios(port_montecarlo,max_sharpe,tickers)
