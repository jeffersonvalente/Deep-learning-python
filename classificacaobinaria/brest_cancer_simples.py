import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score


#--------------------------------------------- carregando a base de dados ---------------------------------------------#

#previsores
X = pd.read_csv('classificacaobinaria/entradas_breast.csv')

#classe
y = pd.read_csv('classificacaobinaria/saidas_breast.csv')

#faz a divisão dos dados para treinamento (test_size=0.25%) = 25% para teste e 75% para treinamento
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25)

#--------------------------------------------- criando estrutura da rede neural ---------------------------------------------#

rede_neural = Sequential(
    [
        tf.keras.layers.InputLayer(shape = (30,)), #camada de entrada com 30 neurônios

        #camada oculta com 16 neurônios 30 unidades de entrada + 1 unidade de saída / 2 = 16(arrendondado)
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform' ),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform' ),

        tf.keras.layers.Dense(units = 1, activation = 'sigmoid'), #camada de saída com 1 neurônio
    ]
)

rede_neural.summary()   
#--------------------------------------------- compilando a rede neural ---------------------------------------------#

otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5) #otimizador Adam 
rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy']) #adam = algoritmo de otimização 
                                                                                                     # binary_crossentropy = função de perda para classificação binária
                                                                                                     # loss = função de perda

#--------------------------------------------- treinando a rede neural ---------------------------------------------#

rede_neural.fit(X_treinamento, y_treinamento, batch_size = 10, epochs = 100) #batch_size = 10, 10 registros por vez

#--------------------------------------------- avaliando a rede neural ---------------------------------------------#

previsores = rede_neural.predict(X_teste)
previsores = (previsores > 0.5)

accuracy_score(y_teste, previsores)   #acurácia da rede neural  #verdadeiro positivo + verdadeiro negativo / total
print(accuracy_score(y_teste, previsores))

confusion_matrix(y_teste, previsores) #matriz de confusão da rede neural #verdadeiro positivo, falso positivo, verdadeiro negativo, falso negativo
print(confusion_matrix(y_teste, previsores))

rede_neural.evaluate(X_teste, y_teste) #avaliação da rede neural #loss e acurácia
print(rede_neural.evaluate(X_teste, y_teste))