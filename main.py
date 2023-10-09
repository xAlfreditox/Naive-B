import pandas as pd
import naive as nb

def main():
    data = pd.read_csv('Iris.csv')
    print("Datos con los que estaremos trabajando: \n")
    print(data)
    
    dataTrain = data.sample(frac=0.7, replace=False)
    print(dataTrain)
    tableMedias = nb.tablaMedias(dataTrain)    
    tableDesviacion = nb.tablaDesviacion(dataTrain)
    frecuenciasIris = nb.frecuenciaIris(dataTrain)

    
    
    dataTest = data.drop(dataTrain.index)
    result = nb.naive(dataTest, tableMedias, tableDesviacion,frecuenciasIris)
    print(result) 
    
    pAciertos = nb.aciertos(result)
    print("La cantidad de aciertos fue de: ", pAciertos[0], "\n")
    print("La cantidad de errores fue de: ", pAciertos[1], "\n")
    print("En un total de: ", pAciertos[2], "\n")
    print("Lo que nos da una exactitud de: ", (pAciertos[0]/pAciertos[2])*100)
    
    
    
    
main()