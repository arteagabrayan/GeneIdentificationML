# Importamos Librerias
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



opt = st.sidebar.radio('Seleccione una opción:', ['General data analysis', 'Astrocyte cell analysis'])
# Sección de análisis de datos generales
if opt == 'General data analysis':

    st.title('Análisis de datos generales')
    st.write('A se muestran algunos terminos importantes con su definicion para entender de mejor maneja las siguientes graficas.')


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lugar de las celulas")
        st.write("El **TC** es un tipo de célula que denominamos como perteneciente al tejido tumoral del glioblastoma.")
        st.write("El **TP** es un tipo de célula perteneciente a la periferia del tumor, limite entre el tejido tumoral y tejido sano del cerebro.")
        st.write("El **NP** son de células sanas que se encuentran al rededor del tumor.")

    with col2:

        st.subheader("Tipo de celula que se encuentran en el dataset")
        cell_type = ['Astrocytes', 'endothelial', 'microglia', 'neurons', 'oligodendrocytes', 'unpanned']
        selected_dataset = st.selectbox('Select a data set:', cell_type)
        
        if selected_dataset == 'Astrocytes':
            st.write("Definicion de celula............")
        elif selected_dataset == 'endothelial':
            st.write("Definicion de celula............")
        elif selected_dataset == 'microglia':
            st.write("Definicion de celula............")
        elif selected_dataset == 'neurons':
            st.write("Definicion de celula............")
        elif selected_dataset == 'oligodendrocytes':
            st.write("Definicion de celula............")
        elif selected_dataset == 'unpanned':
            st.write("Definicion de celula............")
        

    
    st.title('Análisis de datos generales')

    # Contenido de la página principal
    st.write('En esta aplicación, puede seleccionar una opción en la barra lateral para comenzar.')


    # Creamos la estructura de los datos

    cell_type_TC = ['Astrocytes', 'endothelial', 'microglia', 'neurons', 'oligodendrocytes', 'unpanned']
    cant_cell_TC = [386,94,3,207,4,335]
    cell_cant_TC = pd.DataFrame(cant_cell_TC,cell_type_TC)
    cell_type_TP = ['Astrocytes', 'endothelial', 'microglia', 'neurons']
    cant_cell_TP = [58,1,1,2]
    cell_type_NP = ['Astrocytes', 'microglia', 'neurons', 'oligodendrocytes', 'unpanned']
    cant_cell_NP = [258,510,283,63,13]

    # Creamos una lista con los nombres de los conjuntos de datos y una lista con las cantidades correspondientes
    datasets = {'TC': [cell_type_TC, cant_cell_TC], 'TP': [cell_type_TP, cant_cell_TP], 'NP': [cell_type_NP, cant_cell_NP]}

    # Creamos un desplegable para que el usuario pueda seleccionar el conjunto de datos a graficar
    selected_dataset = st.selectbox('Select a data set:', list(datasets.keys()))

    # Obtenemos los datos del conjunto seleccionado
    data = datasets[selected_dataset]


    
    # Creamos la figura y el eje
    fig, ax = plt.subplots()

    # Creamos el histograma en el eje
    ax.bar(data[0], data[1])

    # Agregamos etiquetas al eje x e y y un título
    ax.set_xlabel('Cell type')
    ax.set_ylabel('number of cells')
    ax.set_title(f'Number of cells per type in {selected_dataset}')

    # Mostramos la gráfica en la página web
    st.pyplot(fig)



###########################################################################

    # Mostramos grafica de Torta de como se distribuyen los datos por tipo de celula

    st.title('Diagrama de Torta')
    st.write('Este diagrama muestra la distribucion de datos por tipo de celula.')

    # Creamos la estructura de los datos

    cant_Astrocytes = {'TC':386,'TP':58,'NP':258}
    cant_endothelial = {'TC':94,'TP':1}
    cant_microglia = {'TC':3,'TP':1,'NP':510}
    cant_neurons = {'TC':207,'TP':2,'NP':283}
    cant_oligodendrocytes = {'TC':4,'NP':63}
    cant_unpanned = {'TC':335,'NP':13}
    
    # Obtener las claves de las listas como opciones para el selector
    opciones = ['Astrocytes', 'endothelial', 'microglia', 'neurons', 'oligodendrocytes', 'unpanned']
    opcion_seleccionada = st.selectbox('Selecciona una lista:', opciones)

    # Obtener los datos de la lista seleccionada
    datos = []
    if opcion_seleccionada == 'Astrocytes':
        datos = cant_Astrocytes
    elif opcion_seleccionada == 'endothelial':
        datos = cant_endothelial
    elif opcion_seleccionada == 'microglia':
        datos = cant_microglia
    elif opcion_seleccionada == 'neurons':
        datos = cant_neurons
    elif opcion_seleccionada == 'oligodendrocytes':
        datos = cant_oligodendrocytes
    else:
        datos = cant_unpanned

    # Crear la gráfica de torta con los datos
    fig, ax = plt.subplots()
    ax.pie(datos.values(), labels=datos.keys(), autopct='%1.1f%%')
    ax.set_title(f'Gráfica de torta de {opcion_seleccionada}')
    st.pyplot(fig)




# Sección de análisis de células astrocitarias
else:
    st.title('Análisis de células astrocitarias')
    st.write('Aquí puede incluir todas las herramientas y gráficos de análisis de células astrocitarias que desee.')

