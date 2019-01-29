# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:40:57 2018

@author: adrisanchez
@ adrisanchez@correo.ugr.es
@ 76655183R
"""
import vrep
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from random import sample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


vrep.simxFinish(-1) #Terminar todas las conexiones
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) #Iniciar una nueva conexion en el puerto 19999 (direccion por defecto)

def get_data (own = False):
    '''
    Función para recoger datos laser del simulador V-REP y crear un archivo
    asociado con los puntos detectados.
    '''
    
    if (own):
        name = input("Introduce el nombre del fichero: ")
        
        tiempo = int(input("Introduce el intervalo de tiempo (s): "))
        
        iteraciones = int(input("Introduce el número de iteraciones: "))
    else:
        name = "laser.dat"
        tiempo = 1
        iteraciones = 1
        
    # Abrimos el fichero para escritura
    path = "data/{}".format(name)
    fd = open(path, 'w')
    
    for i in range(0, iteraciones):
        
        #listas para recibir las coordenadas x, y de los puntos detectados por el laser
        puntosx=[] 
        puntosy=[]
        
        returnCode, signalValue = vrep.simxGetStringSignal(clientID,'LaserData',vrep.simx_opmode_buffer) 
        
        time.sleep(tiempo) #esperamos un tiempo para que el ciclo de lectura de datos no sea muy rápido
        
        datosLaser=vrep.simxUnpackFloats(signalValue)
        
        for indice in range(0,len(datosLaser),3):
            puntosx.append(datosLaser[indice+1])
            puntosy.append(datosLaser[indice+2])
                
        plt.clf()    
        plt.plot(puntosx, puntosy, 'r.')
        
        plt.axis([0, 4, -2, 2])  
        
        plt.show()
        
        datos = "Iteracion {}_{}".format(i, len(puntosx))
        
        for x, y in zip(puntosx, puntosy):
            datos += " " + str(x) + " " + str(y)
            
        datos += "\n"
        
        fd.write(datos)
    
    fd.close()
        

def define_cluster (nmin = 6, nmax = 51, umbral = 0.03, own = False):
    '''
    Función que agrupa los puntos leídos mediante laser en clústeres.
    Para ello, hay que especificar el nombre de los ficheros a leer.
    '''
    
    ficheros = []
    
    if (own):
        print("Introduce el nombre de los ficheros a leer. Escribe 'fin' para " +
             "finalizar la lectura")
        
        path = input("--> ")
        
        # Guardamos los ficheros
        while (path != "fin"):
            ficheros.append(path)
            path = input("--> ")
    else:
        ficheros.append("laser.dat")
        
    # Creamos un array vacío
    puntos = np.array([])
    
    for file in ficheros:
        path = "data/{}".format(file)
        
        fd = open(path, 'r')
        
        # Leemos línea por línea
        for line in fd:
            lista = line.split()
            
            del lista[0:2]
            
            # Convertimos los string a float
            lista_fl = list(map(lambda x: float(x), lista))
            
            # Insertamos los puntos
            puntos = np.append(puntos, lista_fl)
            
        # Cerramos el fichero
        fd.close()
        
    
    # Cambiamos el shape del vector, ahora la primera columna será la componente
    # x de los puntos y la segunda columna la componente y.
    fil = int(puntos.shape[0] / 2)
    puntos.shape = (fil, 2)
    
    # Abrimos el archivo para realizar la escritura
    if (own):
        name = input("Introduzca el nombre del archivo: ")
    else:
        name = "cluster.dat"
        
    path = "clusters/{}".format(name)
    
    fd = open(path, 'w')
    
    cluster = 0
    
    lista = []
    
    anteriores = True
    
    for coord in puntos:
        
        x = coord[0]
        y = coord[1]
     
        if (anteriores):
            x_ant = x
            y_ant = y
            
            anteriores = False
            
        # Calculamos la distancia
        dist = np.sqrt( (x - x_ant)**2 + (y - y_ant)**2 )
        
        # Si la distancia es menor al umbral y todavía no se ha completado el clúster
        if (dist <= umbral and len(lista) < nmax):
            
            # Incorporamos a la lista de puntos los actuales
            lista.extend(coord)
            
            # Actualizamos las coordenadas de los puntos anteriores
            x_ant = x
            y_ant = y
        
        # En caso contrario, finalizamos el clúster actual
        else:
            
            # Comprobamos si el clúster llega al mínimo, en caso afirmativo, 
            # aceptamos el clúster
            if (len(lista) > nmin):
                cadena = str(cluster) + " " + str(len(lista))
                
                for pt in lista:
                    cadena += " " + str(pt)
                
                cadena += "\n"
                
                # Escribimos la información al fichero
                fd.write(cadena)
                
            # Rechazamos el clúster
            else:
                anteriores = True
                
            cluster += 1
            lista = []
            
    fd.close()
    
def cluster_geometric (own = False):
    '''
    Función para calcular las características geométricas de los clústeres.
    '''
    
    if (own):
        # Cargamos el fichero que contiene los datos del clúster
        name = input("Introduzca el nombre del fichero (clusters): ")
        path = "clusters/{}".format(name)
        
        fd = open(path, 'r')
        
        # Abrimos el fichero para la lectura
        name = input("Introduce el nombre del nuevo fichero (características): ")
        path = "clusters/{}".format(name)
        
        fc = open(path, 'w')
        
        # Introducimos si el fichero es o no pierna
        pierna = int(input("¿Es fichero de piernas?: (0/1) "))
    else:
        fd = open("clusters/cluster.dat", 'r')
        fc = open("clusters/caracteristicas.dat", 'w')
    
    # Leemos línea por línea
    for line in fd:
        lista = line.split()
        lista = list(map(lambda x: float(x), lista))
        # Guardamos el número de clúster
        n_cluster = int(lista[0])
        
        # Nos quedamos únicamente con los puntos
        del lista[0:2]
        
        # Función lambda para la distancia
        dist = lambda x, y, x_ant, y_ant: np.sqrt( (x - x_ant)**2 + (y - y_ant)**2 )
    
        # Calculamos la anchura
        x = lista[0]
        y = lista[1]
        
        x_2 = lista[len(lista) - 2]
        y_2 = lista[len(lista) - 1]
        
        anchura = dist(x, y, x_2, y_2)
        
        # Guardamos los puntos como arrays de numpy para calcular la profundidad
        P1 = np.array([x, y])
        P2 = np.array([x_2, y_2])
        
        # Inicializamos el perímetro y la profundidad
        perimetro = 0
        profundidad = 0
               
        # Iteramos la lista para calcular el perímetro y la profundidad
        for i in range(2, len(lista), 2):
            
            x   = lista[i-2]
            y   = lista[i-1]
            x_2 = lista[i]
            y_2 = lista[i+1]
                        
            distancia = dist(x, y, x_2, y_2)
            
            perimetro += distancia
            
            P3 = np.array([x, y])
            
            # Por último, calculamos la profundidad
            d = np.abs(np.cross(P2 - P1, P1 - P3)) / np.linalg.norm(P2 - P1)
            
            if (d > profundidad):
                profundidad = d
            
        if (own):
            cadena = "{} {} {} {} {}".format(n_cluster, perimetro, profundidad, anchura, pierna)
        else:
            cadena = "{} {} {} {}".format(n_cluster, perimetro, profundidad, anchura)
        
        cadena += "\n"
        
        fc.write(cadena)
        
    fd.close()
    fc.close()
    
def train_test (train = 0.75, test = 0.25):
    '''
    Función para generar fichero de entrenamiento del clasificador y fichero de 
    prueba a partir de las características extraídas de los clústeres.
    
    '''   
    print("Introduce el nombre de los ejemplos positivos (Piernas)")
    
    path = input("--> ")
    path = "clusters/{}".format(path)
    
    # Abrimos el archivo de características para las piernas.
    fd = open(path, 'r')
    
    # Creamos una lista vacía
    Piernas = []
    
    for line in fd:
        Piernas.append(line)
    
    # Cerramos el fichero
    fd.close()
    
    
    print("Introduce el nombre de los ejemplos negativos (No piernas)")
    
    path = input("--> ")
    path = "clusters/{}".format(path)
    
    # Abrimos el archivo de características para las no piernas.
    fd = open(path, 'r')
    
    # Creamos una lista vacía
    noPiernas = []
    
    for line in fd:
        noPiernas.append(line)
    
    fd.close()
    
    # Calculamos qué archivo tiene menor longitud
    menor = len(Piernas) if len(Piernas) < len(noPiernas) else len(noPiernas)
 
    # Obtenemos una muestra del tamaño del menor para cada fichero
    Piernas = sample(Piernas, menor)
    noPiernas = sample(noPiernas, menor)
    
    ntrain = int(menor * train)
    ntest  = int(menor * test)
    
    dataTrain = Piernas[0:ntrain] + noPiernas[0:ntrain]
    dataTest  = Piernas[-ntest:] + noPiernas[-ntest:]
    
    
    # Volcamos a un fichero los datos de entrenamiento
    fd = open("ML/train.dat", 'w')
    
    for data in dataTrain:
        fd.write(data)
    
    fd.close()
    
    
    # Volcamos a un fichero los datos de prueba
    fd = open("ML/test.dat", 'w')
    
    for data in dataTest:
        fd.write(data)
    
    fd.close()
    
def train_model(show = False):
    '''
    Función para calcular clasificadores a partir de los data set generados
    '''
    # Nombres de las columnas a asignar
    colnames = ['Num_Cluster', 'Perimetro', 'Profundidad', 'Anchura', 'esPierna']
    
    # Cargamos los datos generados
    train = pd.read_csv("ML/train.dat", names = colnames, sep = ' ')
    test  = pd.read_csv("ML/test.dat",  names = colnames, sep = ' ')
    
    # Separamos los datos de las etiquetas tanto en el train como en el test
    train_data  = train.drop('esPierna', axis = 1)
    train_data  = train_data.drop('Num_Cluster', axis = 1)
    train_label = train['esPierna'] 
    
    test_data = test.drop('esPierna', axis = 1)
    test_data = test_data.drop('Num_Cluster', axis = 1)
    test_label = test['esPierna']

    
    # Utilizamos un Kernel Gaussiano de base Radial
        
    svc = SVC(kernel = 'rbf')
    svc.fit(train_data, train_label)
    
    # Una vez disponemos del clasificador, realizamos la predicción sobre test
    test_pred_svc = svc.predict(test_data)

    # Calculamos la precisión obtenida 
    precision_test_svc = accuracy_score(test_label, test_pred_svc)
    
    if (show):
        print("\n -- Kernel Gaussiano base Radial --")
        print("Acc_test: (TP+TN)/(T+P)  %0.2f" % precision_test_svc)
        
        print("Matriz de confusión Filas: predicción Columnas: clases verdaderas")
    
        print(confusion_matrix(test_label, test_pred_svc))
        
        '''
        La precisión mide la capacidad del clasificador en no etiquetar como positivo un ejemplo que es negativo.
        El recall mide la capacidad del clasificador para encontrar todos los ejemplos positivos.
        '''
        
        print("Precision= TP / (TP + FP), Recall= TP / (TP + FN)")
        print("f1-score es la media entre precisión y recall")
        print(classification_report(test_label, test_pred_svc))
        print("--------------------------------------------------------\n")
    
    # Utilizamos un Kernel Gaussiano de base Radial

    dtree = DecisionTreeClassifier(random_state = 0)
    dtree.fit(train_data, train_label)
    
    # Una vez disponemos del clasificador, realizamos la predicción sobre test
    test_pred_dtree = dtree.predict(test_data)

    # Calculamos la precisión obtenida 
    precision_test_dtree = accuracy_score(test_label, test_pred_dtree)
    
    if (show):
        print("\n -- Árbol de decisión --")
        print("Acc_test: (TP+TN)/(T+P)  %0.2f" % precision_test_dtree)
        
        print("Matriz de confusión Filas: predicción Columnas: clases verdaderas")
    
        print(confusion_matrix(test_label, test_pred_dtree))
        
        '''
        La precisión mide la capacidad del clasificador en no etiquetar como positivo un ejemplo que es negativo.
        El recall mide la capacidad del clasificador para encontrar todos los ejemplos positivos.
        '''
        
        print("Precision= TP / (TP + FP), Recall= TP / (TP + FN)")
        print("f1-score es la media entre precisión y recall")
        print(classification_report(test_label, test_pred_dtree))
    
    if (precision_test_svc < precision_test_dtree):
        return dtree
    else:
        return svc
        
# Función main del fichero    
def main():
   
    if clientID!=-1:
        print ('Conexion establecida')
     
    else:
        sys.exit("Error: no se puede conectar") #Terminar este script
     
    #Guardar la referencia de la camara
    _, camhandle = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)
     
    #acceder a los datos del laser
    _, datosLaserComp = vrep.simxGetStringSignal(clientID,'LaserData',vrep.simx_opmode_streaming)
     
    #Iniciar la camara y esperar un segundo para llenar el buffer
    _, resolution, image = vrep.simxGetVisionSensorImage(clientID, camhandle, 0, vrep.simx_opmode_streaming)
    time.sleep(1)
    
    # Obtenemos los datos mediante laser
    get_data()
    
    # Definimos los clústers a partir de los datos
    define_cluster()
    
    # Calculamos las características geométricas de los clústeres ["clusters/caracteristicas.dat"]
    cluster_geometric()
    
    # Obtenemos el predictor
    ML_pred = train_model()
    
    # Nombres de las columnas a asignar
    colnames = ['Num_Cluster', 'Perimetro', 'Profundidad', 'Anchura']
    
    # Cargamos los datos generados
    dataset = pd.read_csv("clusters/caracteristicas.dat", names = colnames, sep = ' ')
    test_data  = dataset.drop('Num_Cluster', axis = 1)
    
    # Obtenemos la predicción
    test_pred = ML_pred.predict(test_data)
    
    # Leemos los puntos de cada clúster para proceder a pintarlos
    fd = open("clusters/cluster.dat", 'r')
    
	# Tres dimensiones - [x, y, etiqueta (0/1)]
    puntos = np.empty((0,3))
    
    cl = 0
    
    for line in fd:
        cluster = line.split()
        cluster = list(map(lambda x: float(x), cluster))
        
		# Borramos los dos primeros elementos, ya que no son necesarios
        del cluster[0:2]
        
		# Convertimos la lista a un array y modificamos su dimensión
        cluster = np.asarray(cluster)
        cluster.shape = (int(len(cluster)/2), 2)
        
		# Obtenemos la etiqueta predicha para el clúster nº cl
        number = test_pred[cl]
        
        label = np.full((cluster.shape[0], 1 ), number)
        
        cluster = np.concatenate((cluster, label), axis = 1)
        
        puntos = np.concatenate((puntos, cluster), axis = 0)
        cl += 1
        
    fd.close()
    
	# Dibujamos la gráfica final
    plt.scatter(puntos[:, 0], puntos[:, 1], s = 10, c = puntos[:, 2], cmap = 'viridis')
    plt.xlim(0, 4)
    plt.ylim(-2, 2)
    
if __name__ == "__main__":
    main()
    
