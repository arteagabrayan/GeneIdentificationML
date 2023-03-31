# Importamos Librerias
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Brain Study", page_icon="./images/logo.png", layout="centered") #layout="wide")#layout="centered")

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
.css-9s5bis.edgvbvh3
{
    visibility: hidden;
}
.css-h5rgaw.egzxvld1
{
    visibility: hidden;
}
</style>
""",unsafe_allow_html=True)

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

st.title("Aplicaciones de aprendizaje automático sobre heterogeneidad intratumoral en glioblastoma utilizando datos de secuenciación de ARN de una sola célula")

opt = st.sidebar.selectbox('Seleccionar Análisis', ['General', 'Astrocitos'])

# Sección de análisis de datos generales
if opt == 'General':
    st.subheader("General")
    st.image("./images/DNAgold.png", caption= "El cerebro, nuestro procesador en la vida.")

    text = "El glioblastoma es un tipo de cáncer cerebral muy agresivo y de rápido crecimiento. Este tumor es 1.58 veces más frecuente en hombres y representa el 47.7% de todos los tumores malignos primarios del sistema nervioso central. El tratamiento del glioblastoma generalmente incluye cirugía, radioterapia y quimioterapia. Tiene un tiempo medio de supervivencia inferior a dos años desde el diagnóstico"
    
    items = text.split(". ")
    for item in items:
        st.write("- " + item + ".")

    st.title('Análisis de datos general')
    st.write('Definición de terminos importantes en esta área de estudio.')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clasificación según el lugar de las células")
        st.write("El núcleo del tumor (**TC**, por sus siglas en ingles - Tumor Core) representa a células que forman parte del tejido tumoral del glioblastoma.")
        st.write("La periferia del tumor (**TP**, por sus siglas en inglés - Tumor Periphery) representa a células pertenecientes a la periferia del tumor, limite entre el tejido tumoral y tejido sano del cerebro.")
        st.write("La periferia normal (**NP**, por sus siglas en inglés - Normal Periphery) representa a células sanas que se encuentran alrededor del tumor.")

    with col2:

        st.subheader("Tipo de células relacionadas")
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
    

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cantidades para la clasificación según el lugar de las células")
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


    with col2:

        ###########################################################################

        # Mostramos grafica de Torta de como se distribuyen los datos por tipo de celula

        st.subheader("Cantidades y porcentajes para el tipo de células relacionadas")
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
        opcion_seleccionada = st.selectbox('Selecciona un tipo de célula:', opciones)
        
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
        #plt.setp(autotexts, size=10, color='gray', weight="bold")

        # Mostrar la gráfica de torta en la página web
        st.pyplot(fig)

# Sección de análisis de células astrocitarias
else:
    st.subheader("Astrocitos")
    st.image("./images/DNAblue.png",caption="El cerebro, nuestro procesador en la vida. ")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tipo de Algoritmo")
        st.write("Definicion del algoritmo implementado para seleccionar biomarcadores mas representativos.")
        algo = ['XGB', 'LR', 'SVM', 'GB', 'RF', 'ET']
        selected_dataset = st.selectbox('Seleccionar un algoritmo de Machine Learning:', algo)
        if selected_dataset == 'XGB':
            st.write("XGBoost (Extreme Gradient Boosting) es un algoritmo de aprendizaje supervisado basado en árboles de decisión. Es una mejora del algoritmo de Gradient Boosting que utiliza una combinación de múltiples árboles de decisión más pequeños para mejorar la precisión de las predicciones. XGBoost utiliza regularización y técnicas de poda de árboles para prevenir el sobreajuste y mejorar la generalización del modelo. Es ampliamente utilizado en competiciones de ciencia de datos y aprendizaje automático debido a su alta precisión y velocidad de entrenamiento.")
        elif selected_dataset == 'LR':
            st.write("LR (Logistic Regression) es un algoritmo de aprendizaje supervisado utilizado para predecir la probabilidad de que un evento ocurra. A diferencia de la regresión lineal, que se utiliza para predecir valores continuos, la regresión logística se utiliza para predecir valores discretos, como la clasificación binaria (0 o 1). La regresión logística utiliza una función logística para modelar la relación entre las variables independientes y la variable dependiente. Esta función convierte cualquier valor de entrada en un valor entre 0 y 1, lo que se puede interpretar como una probabilidad. La regresión logística es comúnmente utilizada en problemas de clasificación binaria y multiclase en áreas como la medicina, finanzas y marketing.")
        elif selected_dataset == 'SVM':
            st.write("SVM (Support Vector Machines) es un algoritmo de aprendizaje supervisado utilizado para problemas de clasificación y regresión. El objetivo de SVM es encontrar el hiperplano que mejor separa las clases de datos. En el caso de la clasificación binaria, el hiperplano es una línea que separa los datos en dos clases. En el caso de la clasificación multiclase, se pueden utilizar varios hiperplanos para separar los datos.")
        elif selected_dataset == 'GB':
            st.write("GB (Gradient Boosting) es un algoritmo de aprendizaje supervisado que se utiliza para hacer predicciones en problemas de regresión y clasificación. Al igual que otros algoritmos de boosting, GB construye un modelo en etapas, utilizando información de los modelos anteriores para mejorar el modelo actual.")
        elif selected_dataset == 'RF':
            st.write("RF (Random Forest) es un algoritmo de aprendizaje supervisado utilizado para problemas de clasificación y regresión. RF se basa en la construcción de múltiples árboles de decisión y combina sus predicciones para obtener una predicción más precisa y estable.En lugar de construir un solo árbol de decisión, RF construye varios árboles de decisión de manera aleatoria a partir de diferentes subconjuntos de los datos y de las características. Cada árbol de decisión se ajusta a una submuestra aleatoria de los datos y de las características, lo que reduce el riesgo de sobreajuste y mejora la generalización del modelo. ")
        elif selected_dataset == 'ET':
            st.write("ET (Extra Trees) es un algoritmo de aprendizaje supervisado utilizado para problemas de clasificación y regresión. Es similar a Random Forest (RF) en que construye múltiples árboles de decisión, pero se diferencia en cómo se construyen y combinan los árboles. En ET, cada árbol se construye utilizando un subconjunto aleatorio de las características del conjunto de datos. A diferencia de RF, donde se realiza una selección de características aleatorias y se evalúa la mejor división, en ET se selecciona una división aleatoria en cada nodo. Esto significa que los árboles individuales en ET son más aleatorios y, por lo tanto, más diversos.")
            
    with col2:
        st.subheader("Escenario")
        escenario = ['1: TP vs TC', '2: TP vs NP', '3: NP vs TPC', '4: TC vs TP vs NP']
        st.write("Los escenarios se utilizaron para entrenar los modelos y asi definir cuales serian los biomarcadores mas representativos apra el glioblastoma.")
        selected_dataset = st.selectbox('Seleccionar un escenario:', escenario)
        
        if selected_dataset == '1: TP vs TC':
            st.write("Se realiza la clasificación de la periferia del tumor (TP) contra el núcleo del tumor (TC)")
        elif selected_dataset == '2: TP vs NP':
            st.write("Se realiza la clasificación de la periferia del tumor (TP) contra la periferia normal (NP)")
        elif selected_dataset == '3: NP vs TPC': 
            st.write("Se realiza la clasificación de la periferia normal (NP) contra la unión de la periferia del tumor (TP) y el núcleo del tumor (TC)")
        elif selected_dataset == '4: TC vs TP vs NP':
            st.write("Se realiza la clasificación independiente del núcleo del tumor (TC), la periferia del tumor (TP) y la periferia normal (NP)")

    ###########################################################
    st.subheader("Genes importantes")
    # Contenido de la página principal
    st.write('Ver los genes más importantes por escenario y algoritmo de machine learning.')

    model_gen = pd.read_csv('model_data.csv', sep=";")

    # Establecer un selector con las opciones de escenario
    escenario = st.selectbox("Selecciona un escenario", ("Escenario 1", "Escenario 2", "Escenario 3", "Escenario 4"))

    # Establecer el multiselector basado en la opción seleccionada
    if escenario == "Escenario 1": 
        ML_models = model_gen.columns[:4]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 2":
        ML_models = model_gen.columns[4:8]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 3":
        ML_models = model_gen.columns[8:12]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 4":
        ML_models = model_gen.columns[12:]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])        

    ###############################################

    st.subheader("Análisis de t-SNE")
    st.write('Los gráficos de t-SNE en conjuntos de 23,368, 65, 12 y 8 genes confirman la heterogeneidad del GBM con respecto a los genes seleccionados.')
    # Creamos una lista con los nombres de los conjuntos de datos y una lista con las cantidades correspondientes
    tSNE_images = {'23.368 Genes':'./images/tSNE23368gen.png', '65 Genes': './images/tSNE65gen.png', '12 Genes': './images/tSNE12gen.png', '8 Genes': './images/tSNE8gen.png'}
    images = [Image.open(tSNE_images[dataset]) for dataset in tSNE_images.keys()]
    labels = [dataset for dataset in tSNE_images.keys()]
    # Crear el panel de 2x2 imágenes
    col1, col2 = st.columns(2)
    with col1:
        st.image(images[0], use_column_width=True)
        st.write(labels[0])
        st.image(images[2], use_column_width=True)
        st.write(labels[2])        
    with col2:
        st.image(images[1], use_column_width=True)
        st.write(labels[1])
        st.image(images[3], use_column_width=True)
        st.write(labels[3])
    
    ###############################################

    st.subheader("Análisis de PCA")
    # Contenido de la página principal
    st.write('Ver el análisis de PCA para una cantidad determinada de genes relevantes en el Glioblastoma.')

    # Creamos la estructura de los datos

    # Creamos una lista con los nombres de los conjuntos de datos y una lista con las cantidades correspondientes
    PCA_images = {'PCA 2D - 12 Genes':'./images/PCA_12_2D.png', 'PCA 3D - 12 Genes':'./images/PCA_12_3D.png','PCA 2D - 8 Genes':'./images/PCA_8_2D.png', 'PCA 3D - 8 Genes':'./images/PCA_8_3D.png'}

    # Creamos un desplegable para que el usuario pueda seleccionar el conjunto de datos a graficar
    selected_dataset = st.selectbox('Selecciona la cantidad de genes a los cuales desea aplicar una visualizacion con PCA:', list(PCA_images.keys()))

    # Cargar la imagen correspondiente a la opción seleccionada
    image = Image.open(PCA_images[selected_dataset])
    st.image(image, caption='Análisis de Componentes Principales.')
    
    st.subheader("Biomarcadores")
    st.write("Son genes que se han identificado como expresados de manera diferencial en las células tumorales del glioblastoma en comparación con las células no tumorales.")
    
    # Leer archivo CSV
    df = pd.read_csv('gene_data.csv', sep=':', header=None)
    dict_genes = {}
    for index, row in df.iterrows():
        key = row[0].strip("'")
        value = row[1].strip().strip("'")
        dict_genes[key] = value
    
    # Ordenar alfabeticamente
    dict_genes_sorted = dict(sorted(dict_genes.items(), key=lambda x: x[0]))

    # Indices
    gen_type = list(dict_genes_sorted.keys())
    selected_gen = st.selectbox('Seleccionar un gen:', gen_type)
    
    # Ciclo for para buscar el nombre del gen en el diccionario
    if selected_gen in gen_type:
        st.write(dict_genes_sorted[selected_gen])
    else: st.write("No se encontró información para el gen ingresado.")

    centered_button = """
        <div style='text-align:center;'>
            <button style='width: 100%;' type='submit'>Gracias</button>
        </div>
    """

    if st.button(label=centered_button):
        st.balloons()