# Importamos Librerias
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ocultar header y footer que vienen por defecto
st.markdown("""
<style>
.css-1rs6os.edgvbvh3
{
    visibility: hidden;
}
.css-cio0dv.egzxvld1
{
    visibility: hidden;
}
</style>
""",unsafe_allow_html=True)

st.title("Aplicaciones de machine learning sobre heterogeneidad intratumoral en glioblastoma utilizando datos de secuenciación de ARN")
st.image("./images/cerebro.jpg",caption="El cerebro, nuestro procesador en la vida. Imagen de kjpargeter en Freepik")

opt = st.sidebar.radio('Seleccione una opción:', ['General data analysis', 'Astrocyte cell analysis'])

# Sección de análisis de datos generales
if opt == 'General data analysis':



    st.title('Análisis de datos generales')
    st.write('Definición de terminos importantes en esta área de estudio.')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clasificación según el lugar de las celulas")
        st.write("El núcleo del tumor (**TC**, por sus siglas en ingles - Tumor Core) representa a células que forman parte del tejido tumoral del glioblastoma.")
        st.write("La periferia del tumor (**TP**, por sus siglas en inglés - Tumor Periphery) representa a células pertenecientes a la periferia del tumor, limite entre el tejido tumoral y tejido sano del cerebro.")
        st.write("La periferia normal (**NP**, por sus siglas en inglés - Normal Periphery) representa a células sanas que se encuentran alrededor del tumor.")

    with col2:

        st.subheader("Tipo de celulas relacionadas")
        cell_type = ['Astrocitos', 'Endoteliales', 'Microglía', 'Neuronas', 'Oligodendrocitos', 'Sin clasificar']
        selected_dataset = st.selectbox('Ver definición de la célula seleccionada:', cell_type)
        
        if selected_dataset == 'Astrocitos':
            st.write("Son células gliales que se encuentran en el cerebro y la médula espinal. Desempeñan funciones importantes en la nutrición, mantenimiento y soporte de las neuronas, así como en la formación de barreras hematoencefálicas.")
        elif selected_dataset == 'Endoteliales':
            st.write("Las células endoteliales son las que recubren los vasos sanguíneos y linfáticos en el cuerpo humano. Se encargan de la regulación del flujo sanguíneo y la permeabilidad vascular.")
        elif selected_dataset == 'Microglía':
            st.write("Son células inmunitarias que se encuentran en el cerebro y la médula espinal. Desempeñan un papel importante en la defensa contra las infecciones, la eliminación de células dañadas y la regulación de la inflamación.")
        elif selected_dataset == 'Neuronas':
            st.write("Son células especializadas en la transmisión de señales nerviosas en el cerebro y el sistema nervioso periférico. Son la unidad fundamental del sistema nervioso y son responsables de funciones como la percepción, el pensamiento y el movimiento.")
        elif selected_dataset == 'Oligodendrocitos':
            st.write("Son células gliales que producen y mantienen la mielina, una sustancia aislante que rodea y protege los axones de las neuronas. La mielina es esencial para la transmisión eficiente de las señales nerviosas.")
        elif selected_dataset == 'Sin clasificar':
            st.write("Son células que no se han clasificado en ninguna de las categorías o que aún no se han analizado para determinar su identidad celular.")
    
    st.subheader("Cantidades para la clasificación según el lugar de las celulas")
    # Contenido de la página principal
    st.write('Ver la cantidad de células en cada lugar donde se encuentran respecto al tumor del Glioblastoma.')

    # Creamos la estructura de los datos

    cell_type_TC = ['Astrocitos', 'Endoteliales', 'Microglía', 'Neuronas', 'Oligodendrocitos', 'Sin clasificar']
    cant_cell_TC = [386,94,3,207,4,335]
    #cell_cant_TC = pd.DataFrame(cant_cell_TC,cell_type_TC)
    cell_type_TP = ['Astrocitos', 'Endoteliales', 'Microglía', 'Neuronas']
    cant_cell_TP = [58,1,1,2]
    cell_type_NP = ['Astrocitos', 'Microglía', 'Neuronas', 'Oligodendrocitos', 'Sin clasificar']
    cant_cell_NP = [258,510,283,63,13]

    # Creamos una lista con los nombres de los conjuntos de datos y una lista con las cantidades correspondientes
    datasets = {'TC': [cell_type_TC, cant_cell_TC], 'TP': [cell_type_TP, cant_cell_TP], 'NP': [cell_type_NP, cant_cell_NP]}

    # Creamos un desplegable para que el usuario pueda seleccionar el conjunto de datos a graficar
    selected_dataset = st.selectbox('Selecciona el lugar de las células para visualizar su distribución:', list(datasets.keys()))

    # Agregamos CSS personalizado para cambiar el color del recuadro a naranja
    st.markdown(
        """
        <style>
        div[data-baseweb="select"]>div:first-child {
            border-color: orange !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Obtenemos los datos del conjunto seleccionado
    data = datasets[selected_dataset]


    
    # Creamos la figura y el eje
    fig, ax = plt.subplots()

    # Creamos el histograma en el eje
    colors = ['#FFC04D' if x == 'Astrocitos' else '#ADD8E6' for x in data[0]]
    ax.bar(data[0], data[1], color=colors)

    # Agregamos etiquetas al eje x e y y un título
    ax.set_xlabel('Tipo de célula')
    ax.set_ylabel('Número de células')
    ax.set_title(f'Número de células por tipo en {selected_dataset}')
    
    # Rotamos los valores del eje X 45 grados
    ax.set_xticklabels(data[0], rotation=45, ha='right')

    # Mostramos la gráfica en la página web
    st.pyplot(fig)



###########################################################################

    # Mostramos grafica de Torta de como se distribuyen los datos por tipo de celula

    st.subheader("Cantidades y porcentajes para el tipo de celulas relacionadas")
    st.write('Este diagrama muestra la distribucion de datos por tipo de celula.')

    # Creamos la estructura de los datos

    cant_Astrocytes = {'TC':386,'TP':58,'NP':258}
    cant_endothelial = {'TC':94,'TP':1}
    cant_microglia = {'TC':3,'TP':1,'NP':510}
    cant_neurons = {'TC':207,'TP':2,'NP':283}
    cant_oligodendrocytes = {'TC':4,'NP':63}
    cant_unpanned = {'TC':335,'NP':13}
    
    # Obtener las claves de las listas como opciones para el selector
    opciones = ['Astrocitos', 'Endoteliales', 'Microglía', 'Neuronas', 'Oligodendrocitos', 'Sin clasificar']
    opcion_seleccionada = st.selectbox('Selecciona una lista:', opciones)
    
    # Obtener los datos de la lista seleccionada
    datos = []
    if opcion_seleccionada == 'Astrocitos':
        datos = cant_Astrocytes
    elif opcion_seleccionada == 'Endoteliales':
        datos = cant_endothelial
    elif opcion_seleccionada == 'Microglía':
        datos = cant_microglia
    elif opcion_seleccionada == 'Neuronas':
        datos = cant_neurons
    elif opcion_seleccionada == 'Oligodendrocitos':
        datos = cant_oligodendrocytes
    else:
        datos = cant_unpanned

    # Para mostrar tanto cantidades como porcentajes
    def pie_chart_label(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    # Crear la gráfica de torta con los datos y los valores absolutos y relativos
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(datos.values(), labels=datos.keys(), colors=['#0EBFE9',"#FFD97D","#60D394"], autopct=lambda pct: pie_chart_label(pct, list(datos.values())))
    ax.set_title(f'Distribución por clases para {opcion_seleccionada}')

    # Modificar las propiedades del texto en la gráfica de torta
    plt.setp(autotexts, size=10, color='gray', weight="bold")

    # Mostrar la gráfica de torta en la página web
    st.pyplot(fig)



# Sección de análisis de células astrocitarias
else:
    st.title('Análisis de células astrocitarias')
    st.write('Aquí puede incluir todas las herramientas y gráficos de análisis de células astrocitarias que desee.')

