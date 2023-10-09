import pandas as pd
import math as m
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

def tablaMedias(data):
    table = data.groupby(['Iris']).mean()
    return table

def tablaDesviacion(data):
    table = data.groupby(['Iris']).std()
    return table

def frecuenciaIris(data): 
    frecuencia = data.groupby(['Iris']).size().div(len(data))
    return frecuencia

def funcionDensidad(media, desviacion, x):
      densidad = norm.pdf(x, media, desviacion)
      return densidad
  
def calculoP(row,medias,desviaciones,frecuencia):
    densidadSLength = funcionDensidad(medias.at['Sepal_Length'], desviaciones.at['Sepal_Length'], row.at['Sepal_Length'])
    densidadSWidth = funcionDensidad(medias.at['Sepal_Width'], desviaciones.at['Sepal_Width'], row.at['Sepal_Width'])
    densidadPLength = funcionDensidad(medias.at['Petal_Length'], desviaciones.at['Petal_Length'], row.at['Petal_Length'])
    densidadPWidth = funcionDensidad(medias.at['Petal_Width'], desviaciones.at['Petal_Width'], row.at['Petal_Width'])
    calculo = densidadSLength*densidadSWidth*densidadPLength*densidadPWidth*frecuencia
    return calculo

def naive(data,tableMedias,tableDesviacion, frecuenciaIris):
    probabilidades = pd.DataFrame(columns=['Probabilidad Setosa', 'Probabilidad Versicolor', 'Probabilidad Virginica', 'Clase esperada', 'Clase estimada'])
    for i in range(len(data.index)):
            pSetosa = calculoP(data.iloc[i], tableMedias.loc['Iris-setosa'], tableDesviacion.loc['Iris-setosa'], frecuenciaIris['Iris-setosa'])
            pVersicolor = calculoP(data.iloc[i], tableMedias.loc['Iris-versicolor'], tableDesviacion.loc['Iris-versicolor'], frecuenciaIris['Iris-versicolor'])
            pVirginica = calculoP(data.iloc[i], tableMedias.loc['Iris-virginica'], tableDesviacion.loc['Iris-virginica'], frecuenciaIris['Iris-virginica'])
            
            values = [pSetosa, pVersicolor, pVirginica]
            valor = pSetosa
            estimado = 0
            for x in range(3):
                if(valor<values[x]):
                    valor = values[x]
                    estimado = x
            if(estimado==0):
                estimado = "Iris-setosa"
            if(estimado == 1):
                estimado = "Iris-versicolor"
            if(estimado == 2):
                estimado = "Iris-virginica"                
            probabilidades.loc[len(probabilidades.index)] = {'Probabilidad Setosa' : pSetosa, 'Probabilidad Versicolor' : pVersicolor, 'Probabilidad Virginica' : pVirginica, 'Clase esperada' : data.iloc[i].at['Iris'], 'Clase estimada' : estimado}          
    return probabilidades
    
def aciertos(data): 
    matrizConfusion = pd.DataFrame(columns = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], index=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    matrizConfusion.loc['Iris-setosa'] = [0,0,0]
    matrizConfusion.loc['Iris-versicolor'] = [0,0,0]
    matrizConfusion.loc['Iris-virginica'] = [0,0,0]
    print(matrizConfusion)
    
    aciertos = 0
    errores = 0
    total = len(data.index)
    
    for i in range(total):
        cEsperada = data.iloc[i].at['Clase esperada']
        cEstimada = data.iloc[i].at['Clase estimada']
        matrizConfusion.loc[cEsperada, cEstimada] = matrizConfusion.loc[cEsperada,cEstimada] + 1
        if( cEsperada == cEstimada):
            aciertos+=1
        else:
            errores+=1
    calculo = [aciertos, errores, total]
    print(matrizConfusion)
    return calculo                    
    

    
    



    