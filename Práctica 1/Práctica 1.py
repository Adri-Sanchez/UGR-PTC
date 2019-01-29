# -*- coding: utf-8 -*-
"""

@author: Adrián Sánchez Cerrillo
@email:  adrisanchez@correo.ugr.es
@DNI:    76655183R

 Práctica 1 .- Población Provincias
"""
import numpy as np
import pylab as plt
import pandas as pd
import csv

#########################################
#     Funciones para cargar ficheros    #
#########################################
# Función para cargar el fichero CSV
def load_CSV (filename, void_start = None, void_end = None):
    """
        Función para cargar un archivo CSV.
            - 'filename' -> path del fichero .csv
            - 'void_start' -> número de líneas a eliminar desde el inicio
            - 'void_end' -> Número de líneas a eliminar desde el final
    """ 
    # Abrimos el archivo y lo leemos
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter = ';')
        
        # Convertimos en lista todo lo leído
        data = list(reader)
        
    # Eliminamos, si fuera proporcionado por parámetro, n o m filas iniciales o finales.
    if (void_start is not None):
        del data [:void_start]
    
    if (void_end is not None):
        del data [-void_end:]
    
    # Devolvemos un array de numpy
    return np.array(data)

def load_comunidades (filename):
    """
        Función para cargar las tablas existentes en páginas html
            - 'filename' -> path del fichero .html
    """
    tablas = pd.read_html(filename)
    
    comunidades = np.array(tablas[0])
    
    # Construimos un diccionario
    diccionario = dict(zip(comunidades[1:, 0], comunidades[1:, 1]))
    
    return diccionario
    
def load_provincias (filename):
    # Leemos el fichero html mediante pandas obteniendo las tablas
    tablas = pd.read_html(filename)
    
    # Almacenamos las tablas
    prov = np.array(tablas[0])
    # Borramos las filas no deseadas
    prov = np.delete(prov, (0, 51), axis = 0)
    
    # Juntamos código y pronvincia
    for p in prov:
        cad_prov = p[2] + ' ' + p[3]
        p[2] = cad_prov
    
    # Borramos la columna de las comunidades autónomas y la última que contiene las provincias
    prov = np.delete(prov, (1, 3), axis = 1)
 
    # Declaramos un diccionario
    dicc = {}
    
    # Recorremos los elementos de provincias
    for n in prov:
        key = n[0]

        # Si la clave ya está, añadimos una nueva entrada a la lista
        if (key in dicc):
            dicc[key] += [n[1]]
        else: # En caso contrario, se crea la entrada del diccionario.
            dicc[key] = [n[1]]
    
    return dicc
    
    
#########################################
#    Funciones para realizar cálculos   #
#########################################
# Función para calcular la variación absoluta y relativa
def calcula_variacion (data, anio_ini, anio_fin):
    """
        Función para calcular la variación relativa y absoluta del crecimiento poblacional
            - 'data' -> Matriz de datos conteniendo únicamente la distribución por años
            - 'anio_ini' -> Año de inicio para los cálculos
            - 'anio_fin' -> Año de finalización
    """
    
    # Obtenemos un vector booleano para conocer la posición de los años establecidos
    inicial = np.isin(data[0, :], str(anio_ini))
    final   = np.isin(data[0, :], str(anio_fin))
    
    # Manejamos una copia de la región para hacer cálculos
    copia = np.copy(data[1:, :])
    copia = copia.astype(float)
    
    # Comprobamos si los años proporcionados son válidos
    if (np.count_nonzero(inicial == True) == 0 or 
        np.count_nonzero(final   == True) == 0):
        
        print("No es posible calcular la variación con los años proporcionados.")
        return 0
    
    # Calculamos los índices
    index_fin, = np.where(inicial)
    index_ini, = np.where(final)
    
    index_fin = index_fin[0] + 1
    index_ini = index_ini[0]
    
    # Creamos dos matrices para contener los datos a devolver
    s = copia.shape
    shape = (s[0], s[1] - 1)
    var_abs = np.zeros(shape)
    var_rel = np.zeros(shape)
    
    for i in range(index_ini, index_fin):
        variacion = copia[:, i] - copia[:, i+1]
        var_abs[:, i] = variacion
        var_rel[:, i] = (variacion / copia[:, i+1]) * 100
        
    return (var_abs, var_rel)
    
def calcula_provincias (data, dicc_pobl, dicc_prov):
    """
        Función que calcula los habitantes por provincia
    """
    shape = data.shape
    
    # Creamos una matriz del tamaño necesitado
    calculo = np.zeros((len(dicc_pobl), shape[1]), dtype=object)
    
    # Obtenemos las llaves del diccionario
    keys = dicc_pobl.keys()
    
    i = 0
    
    # Recorremos las claves del diccionario
    for k in keys:
        com = k + ' ' + dicc_pobl[k]
        provincias = dicc_prov[k]
        
        calculo[i, 0] = str(com) 
        
        for j in range(0, shape[0]):
            
            if (data[j, 0] in provincias):
                calculo[i, 1:] += data[j, 1:].astype(float)  
        
        i += 1
        
    return calculo
    
def best_comunidades (comunidades):
    """
        Función que calcula las 10 mejores comunidades autónomas en base
        al número medio de habitantes entre 2017 y 2010
    """
    shape = comunidades.shape
    
    # Creamos un vector de tamaño fijo
    res = np.zeros((shape[0], 2), dtype = 'object')
    
    # Asignamos a la primera columna el nombre de las comunidades, en la siguiente
    # se almacenará la sumatoria de los valores poblacionales en los distintos años
    for i in range(0, len(comunidades)):
        res[i, 0] = comunidades[i, 0]
        
        res[i, 1] = np.sum(comunidades[i, 1:])
        res[i, 1] = res[i, 1] / (shape[1] - 1)
    
    # Se ordena el array obtenido teniendo en cuenta la segunda columna (número de habitantes medio)
    orden = res[np.argsort(res[:, 1])[::-1]]
    # Se guardan los 10 primeros
    orden = orden[0:10, :]
    
    return orden

def save_plots (comunidades, top):
    # Ejercicio 3
    plt.figure(figsize=(15, 3))
    plt.title("Comparativa por género de CC. AA más pobladas (2017)")
    plt.xlabel("Habitantes")
    plt.ylabel("CC. AA")
    
    dato_h = []
    dato_m = []
    coms   = []
    valores = []
    
    for ccaa in comunidades:
        # Es una de las top-10 en población media        
        if (ccaa[0] in top):
            dato_h.append(ccaa[9])
            dato_m.append(ccaa[17])
            coms.append(ccaa[0])
            valores.append(ccaa[8:0:-1])
            
    valores = np.array(valores)
    X = np.arange(len(top))
    # Mostramos la barra correspondiente 
    plt.bar(X + 0.00, dato_h, color = "b", width = 0.25, alpha=0.5,
            label = 'Hombres')
    plt.bar(X + 0.25, dato_m, color = "r", width = 0.25, alpha=0.5,
            label = 'Mujeres')

    ax = plt.gca()
    ax.tick_params(axis='x', which = 'major', labelsize=6)
    plt.xticks(X+0.125, coms)
    plt.legend(loc = "upper right", shadow = True)

    plt.savefig('img/plot1.png', dpi = 300)
    
    # Ejercicio 4
    plt.figure()
    plt.title("Evolución poblacional CC. AA más pobladas")
    plt.xlabel("Año")
    plt.ylabel("Habitantes")
    
    i = 0
    for valor in valores:
        plt.plot(['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], valor, 
                 label=coms[i])
        i += 1
    
    plt.legend(loc = "upper right", shadow = True, bbox_to_anchor=(1.55, 1.05))
    plt.savefig('img/plot2.png', dpi = 200, bbox_inches='tight')
    
    return 0
    
    # Ejercicio 4
#########################################
#      Funciones para construir web     #
#########################################
# Cabecera estructural de las páginas HTML, incluyendo los estilos a aplicar.
def HTML_head():
    cad = """<!DOCTYPE HTML>
             <html lang="es">
                 <head>
                     <title>Práctica 1</title>
                     <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                     <meta name="description" content="Práctica 1">
                     <style>
                    .footer {
                       position: fixed;
                       bottom: 0;
                       width: 100%;
                       background-color: SeaGreen;
                       color: white;
                       text-align: center;
                    }
                    .clear {clear: both; height:150px;}
                    table, th, td{
                            border: 1px solid blue;
                            padding: 5px;
                    }
                    table.blueTable {
                          border: 0px solid #1C6EA4;
                          background-color: #EEEEEE;
                          width: 100%;
                          text-align: left;
                          border-collapse: collapse;
                        }
                        table.blueTable td, table.blueTable th {
                          border: 1px solid #AAAAAA;
                          padding: 5px 3px;
                        }
                        table.blueTable tbody td {
                          font-size: 13px;
                        }
                        table.blueTable tr:nth-child(even) {
                          background: #D0E4F5;
                        }
                        table.blueTable thead {
                          background: #1C6EA4;
                          background: -moz-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
                          background: -webkit-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
                          background: linear-gradient(to bottom, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
                        }
                        table.blueTable thead th {
                          font-size: 15px;
                          font-weight: bold;
                          color: #FFFFFF;
                        }
                        table.blueTable tfoot {
                          font-size: 14px;
                          font-weight: bold;
                          color: #FFFFFF;
                          background: #D0E4F5;
                          background: -moz-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
                          background: -webkit-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
                          background: linear-gradient(to bottom, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
                          border-top: 2px solid #444444;
                        }
                        table.blueTable tfoot td {
                          font-size: 14px;
                        }
                        table.blueTable tfoot .links {
                          text-align: right;
                        }
                        table.blueTable tfoot .links a{
                          display: inline-block;
                          background: #1C6EA4;
                          color: #FFFFFF;
                          padding: 2px 8px;
                          border-radius: 5px;
                        }
                    </style>
                 </head>"""
                 
    return cad

# Método para cerrar la página HTML, incluyendo el footer o pie de página con info. personal
def HTML_close():
    cad = """</body>
        <div class="clear"></div>    
        <div class="footer">
                 <p> Realizado por: Adrián Sánchez Cerrillo </p>
                 <p> DNI: 76655183R </p>
                 <p> Información de contacto: <a href="mailto:adrisanchez@correo.ugr.es">
                     adrisanchez@correo.ugr.es</a></p>
             </div>
        </body></html>"""
    
    return cad
    
def HTML_table(cabecera, data):
    src = "<tr>"
    
    # Recorremos los elementos de la cabecera y los añadimos mediante etiquetas HTML
    for elem in cabecera:
        src += """<th bgcolor="#2182C2">{}</th>""".format(elem)
    
    src += "</thead><tbody>"
    
    # Construimos la tabla 
    for elem in data:
        src += "<tr>"
        
        for i in elem:
            # Excepción para redondear solo aquellos valores que sean numéricos
            try:
                src += """<td>{}</td>""".format(round(float(i), 3))
            except ValueError:
                src += """<td>{}</td>""".format(i)
            
        src += "</tr>"
    
    src += "</table>"
    
    return src

def HTML_ej1 (data, var):
    cad = "<body>"
    
    copia = np.copy(data[1:, 0:8])
    
    # Creamos un vector que contendrá los años para cada tipo de variación
    cabecera = copia[0, 1:]
    cabecera = np.concatenate((cabecera, cabecera), axis = None)
    
    # Creamos una submatriz de los los datos
    # Contendrá las etiquetas y las variaciones
    datos = np.column_stack((copia[1:, 0], var[0], var[1]))
    
    cad += HTML_head()
    cad += """<p><a href="../Index.html"> Volver al menú principal </a></p>
            <br>
            <table class = "blueTable"><thead><tr>
            <th rowspan="2" title></th>
            <th colspan ="7" title="Variación absoluta">Variación absoluta</th>
            <th colspan ="7" title="Variación relativa">Variación relativa</th>
            </tr>
            """
    cad += HTML_table(cabecera, datos)
    cad += HTML_close()
    
    return cad

def HTML_ej2 (data, cabecera):
    cad = "<body>"
    cad += HTML_head()
    
    cad +="""<p><a href="../Index.html"> Volver al menú principal </a></p>
            <br>
            <table class="blueTable"><thead><tr>
            <th rowspan="2" title></th>
            <th colspan ="8" title="Total">Total</th>
            <th colspan ="8" title="Hombres">Hombres</th>
            <th colspan ="8" title="Mujeres">Mujeres</th>
            </tr>
            """
            
    cad += HTML_table(cabecera, data)
    cad += """<br>
            <center><img src="../img/plot1.png" align="bottom" 
                     width="90%" height="40%"></center>"""
    cad += HTML_close()
    
    return cad

def HTML_ej4 (data, cabecera, var_h, var_m):
    cad = "<body>"
    cad += HTML_head()
    
    # Concatenamos la cabecera para tenerla repetida
    cabecera = np.concatenate((cabecera, cabecera, cabecera, cabecera), axis = None)
    
    # Mostramos la cabecera personalizada del ejercicio
    cad +="""<p><a href="../Index.html"> Volver al menú principal </a></p>
            <br>
            <table class="blueTable"><thead><tr>
            <th rowspan="3" title></th>
            <th colspan ="14" title="Variación absoluta">Variación absoluta</th>
            <th colspan ="14" title="Variación relativa">Variación relativa</th>
            </tr>
            <tr>
            <th colspan ="7" title="Hombres" bgcolor="#2182C2" >Hombres</th>
            <th colspan ="7" title="Mujeres" bgcolor="#2182C2" >Mujeres</th>
            <th colspan ="7" title="Hombres" bgcolor="#2182C2">Hombres</th>
            <th colspan ="7" title="Mujeres" bgcolor="#2182C2">Mujeres</th>
            </tr>
            """
    
    # Creamos una submatriz de los los datos
    # Contendrá las etiquetas y las variaciones
    datos = np.column_stack((data[:, 0], var_h[0], var_m[0], 
                                          var_h[1], var_m[1]))
    
    # Llamamos a la función que crea una tabla a partir de los datos y su cabecera
    cad += HTML_table(cabecera, datos)
    cad += """<br>
            <center><img src="../img/plot2.png" align="bottom" 
                     width="50%" height="40%"></center>"""
    cad += HTML_close()
    
    return cad

def HTML_index():
    cad = ""
    cad += HTML_head() # Imprimimos la cabecera
    # Añadimos el cuerpo de la página
    cad += """<body bgcolor="ACE0E5">
    
             <center><h1>Instituto Nacional de Estadística.</h1>
                     <img src="img/etsiit.jpg" align="bottom">
             </center>
             <h2>Práctica 1</h2>
             <ul>
                 <li>
                     <a href="html/variacionProvincias.html"> Variación por Provincias </a>
                 </li>
                 <li>
                     <a href="html/poblacionComAutonomas.html"> Población por Comunidades Autónomas </a>
                 </li>
                 <li>
                     <a href="html/variacionComAutonomas.html"> Variación por Comunidades Autónomas </a>
                 </li>
             </ul>
             """
             
    # Añadimos los elementos que cierran la página
    cad += HTML_close()
             
    return cad

def HTML_write (filename, content):
    fd = open(filename, 'wb')
    fd.write(content.encode('utf-8'))
    fd.close()
    

#########################################
#        Ejecución de la práctica       #
#########################################  

# ---- Carga de archivos ----    
# Cargamos el fichero .csv
csv_pobl = load_CSV("data/poblacionProvinciasHM2010-17.csv", 4, 4)

# Cargamos en un diccionario las comunidades autónomas y sus códigos
dic_comu = load_comunidades("data/comunidadesAutonomas.htm")

# Ahora las provincias
dic_prov = load_provincias('data/comunidadAutonoma-Provincia.htm')


# ---- Cálculos ----
# Calculamos las variaciones para el ejercicio 1
variations = calcula_variacion(csv_pobl[1:, 1:9], '2011', '2017')

# Calculamos el número de habitantes por comunidad autonoma
hab_comunidad = calcula_provincias(csv_pobl, dic_comu, dic_prov)

# Calculamos las 10 mejores comunidades autónomas en media de población
top_comunidad = best_comunidades(hab_comunidad[:, 0:9])

# Guardamos la gráfica del ejercicio 3
save_plots(hab_comunidad, top_comunidad[:, 0])

# Calculamos las variaciones para el ejercicio 4
aux = np.row_stack((csv_pobl[1, 1:9], hab_comunidad[:, 9:17]))
var_H = calcula_variacion(aux, '2011', '2017')
aux = np.row_stack((csv_pobl[1, 1:9], hab_comunidad[:, 17:]))
var_M = calcula_variacion(aux,  '2011', '2017')


# ---- Escritura HTML ----
# Creamos el índice de las páginas 
HTML_write("Index.html", HTML_index())

# Creamos el .html del ejercicio 1
HTML_write("html/variacionProvincias.html", HTML_ej1(csv_pobl, variations))

# Creamos el .html del ejercicio 2
HTML_write("html/poblacionComAutonomas.html", HTML_ej2(hab_comunidad, csv_pobl[1, 1:]))

# Creamos el .html del ejercicio 4
HTML_write("html/variacionComAutonomas.html", HTML_ej4(hab_comunidad, csv_pobl[1, 1:8],
                                                       var_H, var_M))