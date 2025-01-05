import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin

#--------------------------------------------- carregando a base de dados ---------------------------------------------#

X = pd.read_csv('classificacaobinaria/entradas_breast.csv')     #previsores
y = pd.read_csv('classificacaobinaria/saidas_breast.csv')    #classes

#--------------------------------------------- criando estrutura da rede neural ---------------------------------------------#

def criar_rede():
    k.clear_session()
    rede_neural = Sequential(
    [
        tf.keras.layers.InputLayer(shape = (30,)), #camada de entrada com 30 neurônios

        #camada oculta com 16 neurônios 30 unidades de entrada + 1 unidade de saída / 2 = 16(arrendondado)
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform' ),
        tf.keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform' ),

        tf.keras.layers.Dense(units = 1, activation = 'sigmoid'), #camada de saída com 1 neurônio
    ]
        )
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5) #otimizador Adam
    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy']) # adam = algoritmo de otimização
                                                                                                             # binary_crossentropy = função de perda para classificação binária
                                                                                                             # loss = função de perda
    return rede_neural

#--------------------------------------------- avaliando a rede neural ---------------------------------------------#  
rede_neural = KerasClassifier(model = criar_rede, epochs = 100, batch_size = 10)  #epochs = 100, batch_size = 10 registros por vez
resultados = cross_val_score(estimator = rede_neural, X = X, y = y, cv = 10, scoring = 'accuracy') #cross_val_score = validação cruzada