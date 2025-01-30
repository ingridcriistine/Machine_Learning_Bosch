import pandas as pd
from sklearn.model_selection import train_test_split

global Y
global X

df = pd.read_csv('day.csv')
Y = df['cnt']
X = df.drop('cnt', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.1, 0.2, 0.3, 0.4, 0.5]

def error(w):
    sum = 0
    sumW = 0
    modW = 0
    b = 0.1
    p = 0.2
    s = 0.3

    for linha in range(len(X)):
        for coluna in range(len(X[linha])):
            multVet = X[linha][coluna] * w[coluna]
        
        result = (multVet + b - Y[linha])**2
        sum += result
    
    for item in w:
        sumW += (item ** 2)
        modW += abs(sumW)

    multP = p * sumW
    sig = s * modW

    return result + multP + sig
    
def derivative(w, index):
    err = error(w) 
    
    w[index] += 0.1 # Um passo numa variável especifica em um intervalo de 0.1
    nerr = error(w) # erro no ponto futuro
    w[index] -= 0.1 # Reverte o passo
    
    derivative = (nerr - err) / 0.1 # final menos inicial divido pelo intervalo
    
    return derivative # derivada do erro em relação a variável

# def sgd():

print(error)