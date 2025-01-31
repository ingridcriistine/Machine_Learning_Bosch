import pandas as pd
from sklearn.model_selection import train_test_split

global Y
global X
global w

df = pd.read_csv('day.csv')
Y = df['cnt']
X = df.drop(['dteday', 'cnt'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
b = 0.1

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.1, 0.2, 0.3, 0.4, 0.5]

def error(w, b):
    sum = 0
    sumW = 0
    modW = 0
    p = 0.2
    s = 0.3

    for linha in range(len(X)):
        multVet = 0

        for coluna in range(len(X.iloc[linha])):
            multVet += X.iloc[linha, coluna] * w[coluna]
        
        result = (multVet + b - Y.iloc[linha])**2
        sum += result
    
    for item in w:
        sumW += (item ** 2)
        modW += abs(sumW) 

    multP = p * sumW
    sig = s * modW

    return sum + multP + sig
    
def derivativeW(w, index):
    err = error(w, b) 
    
    w[index] += 0.1 # Um passo numa variável especifica em um intervalo de 0.1
    nerr = error(w, b) # erro no ponto futuro
    w[index] -= 0.1 # Reverte o passo
    
    derivative = (nerr - err) / 0.1 # final menos inicial divido pelo intervalo
    
    return derivative # derivada do erro em relação a variável

def derivativeB(b):
    err = error(w, b) 
    
    b += 0.1 # Um passo numa variável especifica em um intervalo de 0.1
    nerr = error(w, b) # erro no ponto futuro
    b -= 0.1 # Reverte o passo
    
    derivative = (nerr - err) / 0.1 # final menos inicial divido pelo intervalo
    
    return derivative # derivada do erro em relação a variável

alfa = 0.00000000004

def sgdW(w, index):
    return w[index] - alfa * derivativeW(w, index)

def sgdB(b):
    return b - alfa * derivativeB(b)

count = 0
while True:

    if count >= len(X):
        break

    for k in range(101):
        print(error(w, b))
        for i in range(len(w)):
            w[i] = sgdW(w, i)
        b = sgdB(b)
    
    count += 100




