# Importamos Librerias
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Industrias 4.0", page_icon="./images/logo.png", layout="centered") #layout="wide")#layout="centered")

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

st.title("Aplicaciones de machine learning sobre heterogeneidad intratumoral en glioblastoma utilizando datos de secuenciación de ARN")

opt = st.sidebar.radio('Seleccionar Análisis:', ['General', 'Astrocitos'])

# Sección de análisis de datos generales
if opt == 'General':
    st.subheader("General")
    st.image("./images/cerebro.jpg",caption="El cerebro, nuestro procesador en la vida. Imagen de kjpargeter en Freepik")

    text = "El glioblastoma es un tipo de cáncer cerebral muy agresivo y de rápido crecimiento. Este tumor es 1.58 veces más frecuente en hombres y representa el 47.7% de todos los tumores malignos primarios del sistema nervioso central. El tratamiento del glioblastoma generalmente incluye cirugía, radioterapia y quimioterapia. Tiene un tiempo medio de supervivencia inferior a dos años desde el diagnóstico"
    

    items = text.split(". ")
    for item in items:
        st.write("- " + item + ".")
    
    st.image("./images/Tumorcerebral.jpg",caption="Glioblastoma, tumor cerebral agresivo mapeado en detalle genético y molecular. - ALBERT H. KIM - Archivo")
    
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
    st.image("./images/cerebro.jpg",caption="El cerebro, nuestro procesador en la vida. Imagen de kjpargeter en Freepik")

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
        st.subheader("Esenario")
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
        # Muestra un multiselector con las primeras cuatro filas del dataframe
        ML_models = model_gen.columns[:4]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 2":
       # Muestra un multiselector con las primeras cuatro filas del dataframe
        ML_models = model_gen.columns[4:8]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 3":
       # Muestra un multiselector con las primeras cuatro filas del dataframe
        ML_models = model_gen.columns[8:12]
        ML_model_select = st.multiselect("Seleccione un algoritmo:", ML_models.tolist())
        # Muestra las columnas completas con la información seleccionada
        if ML_model_select:
            st.write(model_gen[ML_model_select])
    elif escenario == "Escenario 4":
        # Muestra un multiselector con las primeras cuatro filas del dataframe
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
    
    gen_type = ['ATP1A2', 'NMB', 'SPARCL1', 'USMG5', 'PTN', 'PCSK1N', 'ANAPC11', 'TMSB10', 'TMEM144', 'PSMB4', 'NRBP2', 'FTL',
                'MIR3682', 'S1PR1', 'PRODH', 'SRP9', 'GAP43', 'RPL30', 'LAMA5', 'ECHDC2', 'EGFR', 'CALM1', 'APOD', 'SPOCK1', 'ANXA1', 'PTGDS', 'EIF1', 'VIM', 'MGLL', 'ITM2C', 'PLLP',
                'ITGB8', 'HES6', 'RPS27L', 'GFAP', 'TRIM2', 'APOE', 'ANXA5', 'NAV1', 'TMSB4X', 'HSPB1', 'SEC61G', 'IGSF6', 'IGFBP2', 'RPLP1', 'CSF1R', 'NACA', 'HTRA1', 'CSF3R', 'CREG1', 'FAM107B', 'SLAMF9',
                'GLDN', 'EMP3', 'COMMD6', 'ANXA2', 'RPL38', 'CEBPD', 'APBB1IP', 'HLADRB6', 'TUBGCP2', 'LCP2', 'LOC100505854', 'IFI44', 'GNG11']

    selected_dataset = st.selectbox('Seleccionar un gen:', gen_type)
    
    if selected_dataset == 'ATP1A2':
        st.write("ATP1A2 codifica una subunidad de la bomba de sodio-potasio, que es responsable del mantenimiento del equilibrio de iones en las células. Mutaciones en este gen se han relacionado con migrañas familiares hemipléjicas. No se ha demostrado ninguna relación con glioblastoma u otros cánceres.")
    elif selected_dataset == 'NMB':
        st.write("NMB codifica una proteína relacionada con la hormona liberadora de gastrina. Se ha demostrado que la sobreexpresión de NMB está asociada con la progresión tumoral en varios tipos de cáncer, incluido el glioblastoma.")
    elif selected_dataset == 'SPARCL1':
        st.write("SPARCL1 codifica una proteína matricial extracelular que está involucrada en la adhesión celular y la migración. La baja expresión de SPARCL1 se ha relacionado con la progresión del glioblastoma y otros tipos de cáncer.")
    elif selected_dataset == 'USMG5':
        st.write("USMG5 codifica una proteína mitocondrial que está involucrada en la regulación de la actividad de la cadena de transporte de electrones. No se ha demostrado una relación directa entre USMG5 y el glioblastoma u otros cánceres.")
    elif selected_dataset == 'PTN':
        st.write("PTN codifica una proteína de la familia de las citoquinas que está involucrada en la regulación del crecimiento y la diferenciación celular. La sobreexpresión de PTN se ha relacionado con la progresión tumoral en varios tipos de cáncer, incluido el glioblastoma.")
    elif selected_dataset == 'PCSK1N':
        st.write("PCSK1N codifica una proteína que actúa como inhibidor de la convertasa de proproteína subtilisina/kexina tipo 1 (PCSK1). No se ha demostrado una relación directa entre PCSK1N y el glioblastoma u otros cánceres.")
    elif selected_dataset == 'ANAPC11':
        st.write("ANAPC11 codifica una subunidad del complejo anafase-promoción que está involucrado en la regulación del ciclo celular. Se ha demostrado que la sobreexpresión de ANAPC11 está asociada con la progresión tumoral en varios tipos de cáncer, incluido el glioblastoma.")
    elif selected_dataset == 'TMSB10':
        st.write("TMSB10 codifica una proteína que forma parte del citoesqueleto de actina. Se ha demostrado que la sobreexpresión de TMSB10 está asociada con la progresión tumoral en varios tipos de cáncer, incluido el glioblastoma.")
    elif selected_dataset == 'TMEM144':
        st.write("TMEM144 codifica una proteína transmembrana que se expresa en el retículo endoplásmico. No se ha demostrado una relación directa entre TMEM144 y el glioblastoma u otros cánceres.")
    elif selected_dataset == 'PSMB4':
        st.write("PSMB4 es un gen que codifica para una subunidad del proteasoma 20S, que es una compleja maquinaria intracelular encargada de degradar proteínas no deseadas. Se ha demostrado que la sobreexpresión de PSMB4 está relacionada con la progresión tumoral y la resistencia a la quimioterapia en varios tipos de cáncer, incluyendo el glioblastoma.")
    elif selected_dataset == 'NRBP2':
        st.write("NRBP2 es un gen que codifica para un factor de transcripción implicado en la regulación de la apoptosis y la respuesta a estrés celular. Se ha demostrado que la sobreexpresión de NRBP2 está relacionada con la invasión y la metástasis en varios tipos de cáncer, incluyendo el glioblastoma.")
    elif selected_dataset == 'FTL':
        st.write("FTL es un gen que codifica para la subunidad ligera de la ferritina, una proteína que se encarga de almacenar hierro intracelular. Se ha demostrado que la sobreexpresión de FTL está relacionada con la proliferación celular y la resistencia a la quimioterapia en el glioblastoma.")
    elif selected_dataset == 'MIR3682':
        st.write("MIR3682 es un gen que codifica para un microARN, un tipo de ARN no codificante que regula la expresión génica a nivel post-transcripcional. Se ha demostrado que MIR3682 actúa como un supresor tumoral en el glioblastoma, inhibiendo la proliferación celular y la invasión.")
    elif selected_dataset == 'S1PR1':
        st.write("S1PR1 es un gen que codifica para el receptor de esfingosina-1-fosfato, un mediador de señalización celular implicado en la regulación de procesos fisiológicos y patológicos, como la supervivencia celular, la angiogénesis y la progresión tumoral. Se ha demostrado que la sobreexpresión de S1PR1 está relacionada con la proliferación celular y la invasión en varios tipos de cáncer, incluyendo el glioblastoma.")
    elif selected_dataset == 'PRODH':
        st.write("PRODH es un gen que codifica para la proteína prolina oxidasa, una enzima que cataliza la oxidación de prolina a ácido pirrólido-2-carboxílico. Se ha demostrado que la sobreexpresión de PRODH está relacionada con la apoptosis y la disminución de la viabilidad celular en el glioblastoma.")
    elif selected_dataset == 'SRP9':
        st.write("SRP9 es un gen que codifica para una subunidad de la partícula de reconocimiento de señal, una compleja maquinaria intracelular implicada en la translocación de proteínas hacia el retículo endoplásmico. Aunque no se ha demostrado una relación directa entre SRP9 y el glioblastoma, se ha sugerido que la expresión de SRP9 podría servir como biomarcador de pronóstico en algunos tipos de cáncer.")
    elif selected_dataset == 'GAP43':
        st.write("GAP43 es una proteína que desempeña un papel en el crecimiento y la regeneración de los axones. También se ha encontrado que su expresión está aumentada en ciertos tipos de cánceres, incluyendo el glioblastoma.")
    elif selected_dataset == 'RPL30':
        st.write("RPL30 es un gen que codifica para una proteína ribosómica que forma parte del complejo ribosómico y está involucrado en la síntesis de proteínas. Se ha encontrado que su expresión está disminuida en ciertos tipos de cánceres, incluyendo el glioblastoma.")
    elif selected_dataset == 'LAMA5':
        st.write("LAMA5 es un gen que codifica para una proteína de la matriz extracelular llamada laminina 5. Se ha encontrado que su expresión está aumentada en ciertos tipos de cánceres, incluyendo el glioblastoma, y se ha propuesto que juega un papel en la invasión y la progresión tumoral.")
    elif selected_dataset == 'ECHDC2':
        st.write("ECHDC2 es un gen que codifica para una enoyl-CoA hidratasa que está involucrada en la beta-oxidación de los ácidos grasos. Hasta ahora, no se ha reportado una asociación directa entre ECHDC2 y el glioblastoma.")
    elif selected_dataset == 'EGFR':
        st.write("EGFR es un gen que codifica para el receptor del factor de crecimiento epidérmico (EGFR), que está involucrado en la señalización celular y la regulación del crecimiento y la diferenciación celular. Mutaciones y amplificaciones de EGFR se han encontrado en ciertos tipos de cánceres, incluyendo el glioblastoma, lo que sugiere que puede desempeñar un papel en la progresión tumoral y la resistencia a la terapia.")
    elif selected_dataset == 'CALM1':
        st.write("CALM1 es un gen que codifica para la calmodulina, una proteína reguladora que se une al calcio y está involucrada en la señalización celular. Hasta ahora, no se ha reportado una asociación directa entre CALM1 y el glioblastoma.")
    elif selected_dataset == 'APOD':
        st.write("APOD es un gen que codifica para la apolipoproteína D, una proteína transportadora de lípidos que se ha implicado en varios procesos fisiológicos, incluyendo la respuesta al estrés y la neuroprotección. Hasta ahora, no se ha reportado una asociación directa entre APOD y el glioblastoma.")
    elif selected_dataset == 'SPOCK1':
        st.write("SPOCK1 es un gen que codifica para una proteína de la matriz extracelular llamada SPOCK1 (Sparc/osteonectin, cwcv and kazal-like domains proteoglycan 1). Se ha encontrado que su expresión está aumentada en ciertos tipos de cánceres, incluyendo el glioblastoma, y se ha propuesto que puede desempeñar un papel en la invasión y la progresión tumoral.")
    elif selected_dataset == 'ANXA1':
        st.write("ANXA1 es un gen que codifica para la proteína anexina A1, la cual tiene un papel en la regulación de la respuesta inmune y la inflamación. Además, ANXA1 ha sido implicado en la progresión tumoral y la invasión en varios tipos de cáncer, incluyendo el glioblastoma. Se ha demostrado que la sobreexpresión de ANXA1 en células de glioblastoma se asocia con una mayor capacidad invasiva y una menor supervivencia del paciente.")
    elif selected_dataset == 'PTGDS':
        st.write("PTGDS es un gen que codifica para la prostaglandina D2 sintasa, una enzima que cataliza la producción de prostaglandina D2, una molécula que tiene funciones diversas en el cuerpo, incluyendo la regulación del sueño y la inflamación. En estudios recientes, se ha observado que la expresión de PTGDS está disminuida en glioblastomas, lo que sugiere un posible papel como supresor tumoral.")
    elif selected_dataset == 'EIF1':
        st.write("EIF1 es un gen que codifica para la proteína de iniciación de la traducción 1, la cual juega un papel importante en el proceso de la síntesis de proteínas en la célula. Si bien no se ha identificado una relación directa entre EIF1 y glioblastomas, algunos estudios sugieren que las mutaciones en genes relacionados con la síntesis de proteínas pueden estar implicadas en el desarrollo del cáncer.")
    elif selected_dataset == 'VIM':
        st.write("VIM es un gen que codifica para la vimentina, una proteína que forma parte del citoesqueleto y tiene un papel importante en la estructura y la migración celular. Se ha observado que la expresión de VIM está aumentada en glioblastomas, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'MGLL':
        st.write("MGLL es un gen que codifica para la monoacilglicerol lipasa, una enzima que cataliza la degradación de los lípidos. Si bien no se ha identificado una relación directa entre MGLL y glioblastomas, algunos estudios sugieren que la actividad de la lipasa puede estar implicada en la invasión tumoral.")
    elif selected_dataset == 'ITM2C':
        st.write("ITM2C es un gen que codifica para la proteína integral de membrana 2C, la cual tiene un papel en la regulación de la señalización celular y la apoptosis. Si bien no se ha identificado una relación directa entre ITM2C y glioblastomas, algunos estudios sugieren que la proteína puede tener un papel como supresor tumoral.")
    elif selected_dataset == 'PLLP':
        st.write("PLLP es un gen que codifica para la proteína rica en prolina, leucina y lisina, la cual tiene un papel en la regulación de la señalización celular y la migración. Si bien no se ha identificado una relación directa entre PLLP y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la invasión.")
    elif selected_dataset == 'ITGB8':
        st.write("ITGB8 es un gen que codifica para la subunidad beta 8 de la integrina, una proteína que juega un papel importante en la adhesión celular y la señalización. En estudios recientes, se ha observado que la expresión de ITGB8 está aumentada en glioblastomas, lo que sugiere un posible papel en la progresión tumoral y la invasión. Además, se ha sugerido que la inhibición de ITGB8 podría ser una estrategia terapéutica efectiva para el tratamiento de glioblastomas.")
    elif selected_dataset == 'HES6':
        st.write("HES6 es un gen que codifica para un factor de transcripción que tiene un papel en la regulación del desarrollo embrionario y la diferenciación celular. Si bien no se ha identificado una relación directa entre HES6 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la resistencia a la quimioterapia.")
    elif selected_dataset == 'RPS27L':
        st.write("RPS27L es un gen que codifica para la proteína ribosomal 40S subunidad 27L, la cual tiene un papel en la síntesis de proteínas en la célula. Si bien no se ha identificado una relación directa entre RPS27L y glioblastomas, algunos estudios sugieren que las mutaciones en genes relacionados con la síntesis de proteínas pueden estar implicadas en el desarrollo del cáncer.")
    elif selected_dataset == 'GFAP':
        st.write("GFAP es un gen que codifica para la proteína ácida fibrilar de la gliana, una proteína que se expresa en las células gliales del sistema nervioso central. En glioblastomas, se ha observado que la expresión de GFAP está aumentada, lo que se utiliza como un marcador de diagnóstico para la enfermedad.")
    elif selected_dataset == 'TRIM2':
        st.write("TRIM2 es un gen que codifica para una proteína que tiene un papel en la regulación de la señalización celular y la organización del citoesqueleto. Si bien no se ha identificado una relación directa entre TRIM2 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la invasión.")
    elif selected_dataset == 'APOE':
        st.write("APOE es un gen que codifica para la apolipoproteína E, una proteína que juega un papel importante en el transporte de lípidos en el cuerpo. En estudios recientes, se ha observado que la variante E4 de APOE está asociada con un mayor riesgo de desarrollar glioblastomas, posiblemente debido a su papel en la regulación de la respuesta inflamatoria.")
    elif selected_dataset == 'ANXA5':
        st.write("ANXA5 es un gen que codifica para la proteína anexina A5, la cual tiene un papel en la regulación de la coagulación sanguínea y la apoptosis. Si bien no se ha identificado una relación directa entre ANXA5 y glioblastomas, algunos estudios sugieren que la proteína puede tener un papel como supresor tumoral.")
    elif selected_dataset == 'NAV1':
        st.write("NAV1 es un gen que codifica para el canal de sodio oltage-dependiente Nav1.1, una proteína que tiene un papel importante en la generación y conducción de señales eléctricas en las neuronas. Si bien no se ha identificado una relación directa entre NAV1 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la migración y la invasión celular en otros tipos de cáncer.")
    elif selected_dataset == 'TMSB4X':
        st.write("TMSB4X es un gen que codifica para la tímoseina beta-4, una proteína que tiene un papel en la regulación del citoesqueleto y la migración celular. Si bien no se ha identificado una relación directa entre TMSB4X y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la invasión.")
    elif selected_dataset == 'HSPB1':
        st.write("HSPB1 es un gen que codifica para la proteína de choque térmico 27, la cual tiene un papel en la protección celular contra el estrés y la apoptosis. En glioblastomas, se ha observado que la expresión de HSPB1 está aumentada, lo que sugiere un posible papel en la resistencia a la quimioterapia y la radioterapia.")
    elif selected_dataset == 'SEC61G':
        st.write("SEC61G es un gen que codifica para una subunidad del complejo de proteínas de la membrana del retículo endoplásmico, que tiene un papel en la translocación de proteínas a través de la membrana. Si bien no se ha identificado una relación directa entre SEC61G y glioblastomas, algunos estudios sugieren que las mutaciones en genes relacionados con la síntesis y la translocación de proteínas pueden estar implicadas en el desarrollo del cáncer.")
    elif selected_dataset == 'IGSF6':
        st.write("IGSF6 es un gen que codifica para la proteína de superficie celular inmunoglobulina superfamily member 6, la cual tiene un papel en la adhesión celular y la señalización. Si bien no se ha identificado una relación directa entre IGSF6 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la invasión.")
    elif selected_dataset == 'IGFBP2':
        st.write("IGFBP2 es un gen que codifica para la proteína de unión a IGF-2, la cual tiene un papel en la regulación del crecimiento y la diferenciación celular. En glioblastomas, se ha observado que la expresión de IGFBP2 está aumentada, lo que sugiere un posible papel en la progresión tumoral y la resistencia a la quimioterapia.")
    elif selected_dataset == 'RPLP1':
        st.write("RPLP1 es un gen que codifica para la proteína ribosomal 60S subunidad 41, la cual tiene un papel en la síntesis de proteínas en la célula. Si bien no se ha identificado una relación directa entre RPLP1 y glioblastomas, algunos estudios sugieren que las mutaciones en genes relacionados con la síntesis de proteínas pueden estar implicadas en el desarrollo del cáncer.")
    elif selected_dataset== 'CSF1R':
        st.write("CSF1R es un gen que codifica para el receptor del factor estimulante de colonias de macrófagos 1, la cual tiene un papel en la regulación de la proliferación, diferenciación y supervivencia de los macrófagos. En glioblastomas, se ha observado que la expresión de CSF1R está aumentada, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'NACA':
        st.write("NACA es un gen que codifica para la proteína asociada a la histona H2A-H2B, la cual tiene un papel en la regulación de la expresión génica y la organización de la cromatina. Si bien no se ha identificado una relación directa entre NACA y glioblastomas, algunos estudios sugieren que las mutaciones en genes relacionados con la organización de la cromatina pueden estar implicadas en el desarrollo del cáncer.")
    elif selected_dataset == 'HTRA1':
        st.write("HTRA1 es un gen que codifica para la proteína serina proteasa HTRA1, la cual tiene un papel en la regulación de la señalización celular y la respuesta al estrés. En glioblastomas, se ha observado que la expresión de HTRA1 está disminuida, lo que sugiere un posible papel en la progresión tumoral y la resistencia a la apoptosis.")
    elif selected_dataset == 'CSF3R':
        st.write("CSF3R es un gen que codifica para el receptor del factor estimulante de colonias de granulocitos, la cual tiene un papel en la regulación de la proliferación, diferenciación y supervivencia de los granulocitos. En glioblastomas, se ha observado que la expresión de CSF3R está aumentada, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'CREG1':
        st.write("CREG1 es un gen que codifica para la proteína de crecimiento celular regulado por calcio 1, la cual tiene un papel en la proliferación y diferenciación celular. Si bien no se ha identificado una relación directa entre CREG1 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la invasión en otros tipos de cáncer.")
    elif selected_dataset == 'FAM107B':
        st.write("FAM107B es un gen que codifica para la proteína FAM107B, la cual tiene un papel en la regulación del ciclo celular y la apoptosis. Si bien no se ha identificado una relación directa entre FAM107B y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la resistencia a la quimioterapia en otros tipos de cáncer.")
    elif selected_dataset == 'SLAMF9':
        st.write("SLAMF9 es un gen que codifica para la proteína 9 de la familia de moléculas de señalización y activación de linfocitos, la cual tiene un papel en la regulación de la respuesta inmunitaria. Si bien no se ha identificado una relación directa entre SLAMF9 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación de la respuesta inmunitaria en otros tipos de cáncer.")
    elif selected_dataset == 'GLDN':
        st.write("GLDN es un gen que codifica para la proteína gliomedina, la cual tiene un papel en la adhesión celular y la organización del citoesqueleto. En glioblastomas, se ha observado que la expresión de GLDN está aumentada, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'EMP3':
        st.write("EMP3 es un gen que codifica para la glicoproteína de membrana 3, la cual tiene un papel en la regulación de la adhesión celular y la señalización intracelular. En glioblastomas, se ha observado que la expresión de EMP3 está aumentada, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'COMMD6':
        st.write("COMMD6 es un gen que codifica para la proteína 6 de la familia de proteínas de la comunicación de los dominios, la cual tiene un papel en la regulación de la respuesta inflamatoria y la homeostasis celular. Si bien no se ha identificado una relación directa entre COMMD6 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación de la respuesta inflamatoria en otros tipos de cáncer.")
    elif selected_dataset == 'ANXA2':
        st.write("ANXA2 es un gen que codifica para la proteína anexina A2, la cual tiene un papel en la regulación de la adhesión celular y la señalización intracelular. En glioblastomas, se ha observado que la expresión de ANXA2 está aumentada, lo que sugiere un posible papel en la progresión tumoral y la invasión.")
    elif selected_dataset == 'RPL38':
        st.write("RPL38 es un gen que codifica para la proteína ribosomal L38, la cual tiene un papel en la síntesis de proteínas. Si bien no se ha identificado una relación directa entre RPL38 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la progresión tumoral y la respuesta a la quimioterapia en otros tipos de cáncer.")
    elif selected_dataset == 'CEBPD':
        st.write("CEBPD es un gen que codifica para el factor de transcripción C/EBP delta, la cual tiene un papel en la regulación de la diferenciación celular y la respuesta inflamatoria. En glioblastomas, se ha observado que la expresión de CEBPD está disminuida, lo que sugiere un posible papel en la progresión tumoral y la resistencia a la quimioterapia.")
    elif selected_dataset == 'APBB1IP':
        st.write("APBB1IP es un gen que codifica para la proteína interactuante con la APP y el receptor de lipoproteínas, la cual tiene un papel en la regulación del transporte de proteínas y lípidos. Si bien no se ha identificado una relación directa entre APBB1IP y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación del ciclo celular y la apoptosis en otros tipos de cáncer.")
    elif selected_dataset == 'HLADRB6':
        st.write("HLADRB6 es un gen que codifica para una subunidad del complejo mayor de histocompatibilidad clase II, la cual tiene un papel en la presentación de antígenos a las células T del sistema inmunitario. Si bien no se ha identificado una relación directa entre HLADRB6 y glioblastomas, algunos estudios sugieren que la expresión de los genes del complejo mayor de histocompatibilidad clase II puede estar asociada con una mejor respuesta a la inmunoterapia en algunos tipos de cáncer.")
    elif selected_dataset == 'TUBGCP2':
        st.write("TUBGCP2 es un gen que codifica para la proteína gamma-tubulina complejo 2, la cual tiene un papel en la nucleación y organización de los microtúbulos. Si bien no se ha identificado una relación directa entre TUBGCP2 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación de la mitosis y la progresión tumoral en otros tipos de cáncer.")
    elif selected_dataset == 'LCP2':
        st.write("LCP2 es un gen que codifica para la proteína p85 de la cinasa linfocitaria, la cual tiene un papel en la señalización intracelular y la activación de las células T del sistema inmunitario. Si bien no se ha identificado una relación directa entre LCP2 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación de la respuesta inmunitaria en otros tipos de cáncer.")
    elif selected_dataset == 'LOC100505854':
        st.write("LOC100505854 es un gen cuya función aún no ha sido completamente caracterizada. Si bien no se ha identificado una relación directa entre LOC100505854 y glioblastomas, algunos estudios sugieren que la expresión de genes no codificantes puede estar implicada en la regulación de la expresión génica y la progresión tumoral en algunos tipos de cáncer.")
    elif selected_dataset == 'IFI44':
        st.write("IFI44 es un gen que codifica para la proteína interferón-inducible 44, la cual tiene un papel en la respuesta antiviral y la regulación de la señalización intracelular. Si bien no se ha identificado una relación directa entre IFI44 y glioblastomas, algunos estudios sugieren que la expresión de genes interferón-inducibles puede estar implicada en la regulación de la respuesta inmunitaria y la progresión tumoral en algunos tipos de cáncer.")
    elif selected_dataset == 'GNG11':
        st.write("GNG11 es un gen que codifica para la proteína subunitaria gamma 11 de la proteína G, la cual tiene un papel en la señalización intracelular y la regulación de la actividad de diversas proteínas. Si bien no se ha identificado una relación directa entre GNG11 y glioblastomas, algunos estudios sugieren que la proteína puede estar implicada en la regulación de la angiogénesis y la progresión tumoral en otros tipos de cáncer.")
    else:
        st.write("No se encontró información para el gen ingresado.")

    centered_button = """
        <div style='text-align:center;'>
            <button style='width: 100%;' type='submit'>Gracias</button>
        </div>
    """

    if st.button(label=centered_button):
        st.balloons()