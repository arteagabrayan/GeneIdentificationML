# Importamos Librerias
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Industrias 4.0", page_icon=":guardsman:", layout="centered")#layout="wide")#layout="centered")

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

st.title("Aplicaciones de machine learning sobre heterogeneidad intratumoral en glioblastoma utilizando datos de secuenciación de ARN")

opt = st.sidebar.radio('Seleccionar Análisis:', ['General', 'Astrocitos'])

# Sección de análisis de datos generales
if opt == 'General':
    st.subheader("General")
    st.image("./images/cerebro.jpg",caption="El cerebro, nuestro procesador en la vida. Imagen de kjpargeter en Freepik")

    st.title('Análisis de datos general')
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
    #plt.setp(autotexts, size=10, color='gray', weight="bold")

    # Mostrar la gráfica de torta en la página web
    st.pyplot(fig)



# Sección de análisis de células astrocitarias
else:
    st.subheader("Astrocitos")
    st.image("./images/cerebro.jpg",caption="El cerebro, nuestro procesador en la vida. Imagen de kjpargeter en Freepik")

    col1, col2, col3 = st.columns(3)
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

        st.subheader("Biomarcador")
        gen_type = ['ATP1A2', 'NMB', 'SPARCL1', 'USMG5', 'PTN', 'PCSK1N', 'ANAPC11', 'TMSB10', 'TMEM144', 'PSMB4', 'NRBP2', 'FTL',
                    'MIR3682', 'S1PR1', 'PRODH', 'SRP9', 'GAP43', 'RPL30', 'LAMA5', 'ECHDC2', 'EGFR', 'CALM1', 'APOD', 'SPOCK1', 'ANXA1', 'PTGDS', 'EIF1', 'VIM', 'MGLL', 'ITM2C', 'PLLP',
                    'ITGB8', 'HES6', 'RPS27L', 'GFAP', 'TRIM2', 'APOE', 'ANXA5', 'NAV1', 'TMSB4X', 'HSPB1', 'SEC61G', 'IGSF6', 'IGFBP2', 'RPLP1', 'CSF1R', 'NACA', 'HTRA1', 'CSF3R', 'CREG1', 'FAM107B', 'SLAMF9',
                    'GLDN', 'EMP3', 'COMMD6', 'ANXA2', 'RPL38', 'CEBPD', 'APBB1IP', 'HLADRB6', 'TUBGCP2', 'LCP2', 'LOC100505854', 'IFI44', 'GNG11']
        st.write("Un biomarcador es una medida objetiva de una característica biológica que puede ser utilizada para indicar la presencia, gravedad o progresión de una enfermedad, así como para evaluar la eficacia de un tratamiento.")
        selected_dataset = st.selectbox('Seleccionar un gen:', gen_type)
        
        if selected_dataset == 'ATP1A2':
            st.write("Es el nombre de un gen humano que codifica una proteína llamada subunidad alfa-2 de la ATPasa Na+/K+. Esta proteína se encuentra en la membrana celular y juega un papel importante en la regulación del transporte de iones sodio y potasio a través de la membrana. En algunos tipos de cáncer, como el cáncer de mama y el glioma, se ha encontrado que la expresión del gen ATP1A2 es reducida, lo que sugiere que esta proteína podría tener un papel en la progresión del cáncer,")
        elif selected_dataset == 'NMB':
            st.write("NMB son las siglas de neuromuscular blocking agent, que en español significa agente bloqueador neuromuscular. Estos agentes son sustancias químicas que se utilizan en anestesia para bloquear temporalmente la función de los músculos esqueléticos. En algunos tipos de cáncer, como el cáncer de pulmón, el cáncer de mama y el cáncer de próstata, se ha encontrado que la sobreexpresión del gen NMB se correlaciona con una mayor agresividad y progresión del cáncer. Por lo tanto, NMB se ha propuesto como un posible objetivo terapéutico para el tratamiento del cáncer.")
        elif selected_dataset == 'SPARCL1':
            st.write("Gen humano que codifica una proteína conocida como SPARC-like protein 1. Esta proteína se encuentra en la matriz extracelular de los tejidos y desempeña un papel importante en la regulación del crecimiento y la migración celular, así como en la adhesión celular y la angiogénesis. Se ha demostrado que SPARCL1 se expresa en una variedad de tejidos, incluyendo pulmón, hígado, páncreas, riñón y cerebro, y se ha implicado en la patología de varias enfermedades, incluyendo el cáncer.")
        elif selected_dataset == 'USMG5':
            st.write("El gen codifica una proteína con un papel en el mantenimiento y la regulación de la población de ATP sintasa en la mitocondria. Se ha encontrado que la mutación del gen USMG5 se asocia con un mayor riesgo de desarrollar varios tipos de cáncer, incluyendo el cáncer de mama y el cáncer colorrectal.")
        elif selected_dataset == 'PTN':
            st.write("Es un gen humano que codifica una proteína llamada pleiotrophin. La proteína pleiotrofina es una proteína secretada que se une a los receptores de la superficie celular y está involucrada en diversos procesos biológicos, como el desarrollo del sistema nervioso, la angiogénesis y la reparación de tejidos. Se ha demostrado que la sobreexpresión de PTN está asociada con el cáncer y la inflamación, y puede desempeñar un papel en la progresión y metástasis de algunos tipos de cáncer.")
        elif selected_dataset == 'PCSK1N':
            st.write("Codifica una proteína conocida como proconvertasa subtilisina/kexina tipo 1 inhibitor. Esta proteína es un inhibidor natural de las proconvertasas subtilisinas/kexinas tipo 1 (PCSK1), que son enzimas proteolíticas involucradas en la activación de varias proteínas secretadas, incluyendo hormonas y factores de crecimiento. Se ha demostrado que la mutación del gen PCSK1N se relaciona con un mayor riesgo de desarrollar cáncer colorrectal y que la expresión reducida de PCSK1N se correlaciona con una mayor agresividad y progresión del cáncer de páncreas.")
        elif selected_dataset == 'ANAPC11':
            st.write("Gen humano que codifica una proteína que forma parte de un complejo proteico conocido como anafase-promoting complex/cyclosome (APC/C). El APC/C es una ubiquitina ligasa que regula la progresión del ciclo celular al degradar proteínas clave involucradas en la mitosis y la meiosis. se ha demostrado que la expresión reducida de ANAPC11 se correlaciona con una mayor agresividad y progresión del cáncer de mama y del cáncer de ovario, lo que sugiere que ANAPC11 podría ser un posible biomarcador pronóstico en estos tipos de cáncer.")
        elif selected_dataset == 'TMSB10':
            st.write("Gen humano que codifica una proteína llamada tensina beta-10. La proteína TMSB10 pertenece a una familia de proteínas llamadas tensinas, que se unen a la actina y juegan un papel importante en la regulación del citoesqueleto y la migración celular. Se ha demostrado que la mutación del gen TMSB10 se relaciona con un mayor riesgo de desarrollar cáncer colorrectal y que la expresión reducida de TMSB10 se correlaciona con una mayor agresividad y progresión del cáncer de mama y del cáncer de pulmón. Además, se ha encontrado que la sobreexpresión de TMSB10 se asocia con una mayor supervivencia en pacientes con cáncer de próstata, lo que sugiere que TMSB10 podría ser un posible biomarcador pronóstico en este tipo de cáncer.")
        elif selected_dataset == 'TMEM144':
            st.write("Es el gen humano que codifica una proteína de membrana transmembrana llamada proteína 144 de la membrana transmembrana. Aunque se sabe poco sobre la función exacta de esta proteína, se ha encontrado que se expresa en una variedad de tejidos, incluyendo el cerebro, el hígado y los riñones. se relaciona con un mayor riesgo de desarrollar cáncer colorrectal y que la expresión reducida de TMEM144 se correlaciona con una mayor agresividad y progresión del cáncer de mama y del cáncer de pulmón. ")
        elif selected_dataset == 'PSMB4':
            st.write("Gen humano que codifica una subunidad proteica del proteasoma, una compleja maquinaria proteolítica encargada de degradar proteínas celulares no deseadas o dañadas. La subunidad proteica PSMB4 se encuentra en el núcleo y se une al complejo del proteasoma, ayudando en la degradación de proteínas específicas en el núcleo celular. Se ha demostrado que la mutación del gen PSMB4 se relaciona con un mayor riesgo de desarrollar diversos tipos de cáncer, incluyendo el cáncer de ovario, el cáncer de pulmón y el cáncer de próstata. Además, se ha encontrado que la expresión reducida de PSMB4 se correlaciona con una mayor agresividad y progresión del cáncer de mama y del cáncer de pulmón.")
        elif selected_dataset == 'NRBP2':
            st.write("Codifica una proteína de unión a receptores nucleares conocida como proteína de unión al receptor nuclear relacionada con la inmunidad (NRBP, por sus siglas en inglés). La proteína NRBP2 es una de las dos isoformas producidas por el gen NRBP y se ha demostrado que interactúa con varios receptores nucleares, incluyendo los receptores de andrógenos y estrógenos, y juega un papel importante en la regulación de la transcripción génica. Se ha demostrado que la mutación del gen NRBP2 se relaciona con un mayor riesgo de desarrollar cáncer de pulmón y que la expresión reducida de NRBP2 se correlaciona con una mayor agresividad y progresión del cáncer de mama.")
        elif selected_dataset == 'FTL':
            st.write("Gen humano que codifica la proteína de la ferritina ligera. La ferritina es una proteína esencial para el metabolismo del hierro en las células, que almacena el hierro en forma no tóxica y lo libera cuando es necesario. La proteína FTL forma parte de la ferritina junto con la subunidad pesada codificada por el gen FTH1.  Se ha demostrado que la mutación del gen FTL se relaciona con un mayor riesgo de desarrollar cáncer de hígado y que la expresión reducida de FTL se correlaciona con una mayor agresividad y progresión del cáncer de mama y del cáncer de próstata.")
        elif selected_dataset == 'MIR3682':
            st.write("MicroARN humano, es decir, una pequeña molécula de ARN que regula la expresión génica al unirse a ARN mensajero y promover su degradación o inhibir su traducción. Se ha encontrado que el microARN MIR3682 se expresa en una variedad de tejidos humanos, incluyendo el cerebro, los pulmones y el hígado. Se ha demostrado que MIR3682 actúa como un regulador negativo de varios genes implicados en la proliferación, la invasión y la metástasis celular en el cáncer.")
        elif selected_dataset == 'S1PR1':
            st.write("Gen humano que codifica el receptor 1 de esfingosina-1-fosfato (S1P1), que es un receptor acoplado a proteínas G presente en la superficie de las células. El receptor S1P1 se une a la esfingosina-1-fosfato, un lípido bioactivo, y se ha demostrado que tiene un papel importante en la regulación de la respuesta inmune y la inflamación, así como en la formación de vasos sanguíneos durante el desarrollo embrionario y en la angiogénesis en enfermedades como el cáncer.")
        elif selected_dataset == 'PRODH':
            st.write("Gen humano que codifica la enzima prolina oxidasa, que es responsable de la conversión de prolina a ácido pirroline-5-carboxílico (P5C) en la vía metabólica de la prolina. La prolina es un aminoácido importante que se encuentra en proteínas y también tiene funciones biológicas independientes, como la protección contra el estrés oxidativo. Se ha demostrado que la mutación del gen PRODH se relaciona con un mayor riesgo de desarrollar varios tipos de cáncer, incluyendo el cáncer de mama y el cáncer de pulmón. Además, se ha encontrado que la expresión reducida de PRODH se correlaciona con una mayor agresividad y progresión del cáncer de colon y del cáncer de pulmón.")
        elif selected_dataset == 'SRP9':
            st.write("Gen humano que codifica la proteína 9 de la partícula de reconocimiento de señal (SRP), que es una proteína involucrada en el proceso de translocación de proteínas hacia el retículo endoplásmico durante la síntesis de proteínas. Mutaciones en el gen SRP9 se han relacionado con algunos trastornos genéticos, como el síndrome de Marfan y la ataxia cerebelosa autosómica recesiva tipo 1. Se ha encontrado que la sobreexpresión de SRP9 se asocia con una mayor supervivencia en pacientes con cáncer de pulmón y podría ser un posible biomarcador pronóstico en este tipo de cáncer. Sin embargo, aún se necesita más investigación para entender mejor el papel exacto de SRP9 en el cáncer y su potencial como diana terapéutica.")
        elif selected_dataset == 'GAP43':
            st.write("Codifica la proteína de crecimiento asociada a los axones 43 (GAP-43), que es una proteína importante en el crecimiento y la regeneración de los axones en el sistema nervioso. se ha demostrado que la expresión de GAP-43 aumenta en respuesta a diversas lesiones del sistema nervioso, lo que sugiere que puede ser un objetivo terapéutico para promover la regeneración y la recuperación neuronal en casos de lesiones o enfermedades neurológicas. Si bien GAP43 no se considera un gen directamente relacionado con el cáncer, se ha demostrado que su expresión se encuentra alterada en varios tipos de tumores, incluyendo el cáncer de próstata y el cáncer de mama. ")
        elif selected_dataset == 'RPL30':
            st.write("Gen humano que codifica la proteína ribosomal L30, que es una proteína componente de la subunidad 60S del ribosoma. Mutaciones en el gen RPL30 se han relacionado con algunos trastornos genéticos, como la anemia de Fanconi, aunque se necesita más investigación para comprender mejor la función biológica de RPL30 y cómo se relaciona con la fisiología y la patología humanas. Si bien la mutación en RPL30 no se ha relacionado directamente con el cáncer, se ha encontrado que la expresión de RPL30 está alterada en varios tipos de tumores, incluyendo el cáncer de mama, el cáncer de pulmón y el cáncer de próstata.")
        elif selected_dataset == 'LAMA5':
            st.write("Es un gen humano que codifica la proteína laminina alfa-5, que es una proteína importante en la formación y mantenimiento de la matriz extracelular en diversos tejidos, incluyendo la piel, los músculos y el sistema nervioso.  Mutaciones en el gen LAMA5 se han relacionado con algunos trastornos genéticos, como el síndrome de enfermedad ocular-eritroniquia-arteria pulmonar, aunque se necesita más investigación para comprender mejor la función biológica de LAMA5 y cómo se relaciona con la fisiología y la patología humanas. Se ha demostrado que la sobreexpresión de LAMA5 se asocia con una mayor invasión y metástasis en varios tipos de cáncer, incluyendo el cáncer de mama, el cáncer de pulmón y el cáncer colorrectal.")
        elif selected_dataset == 'ECHDC2':
            st.write("Codifica la proteína enoyl-CoA hydratase domain-containing protein 2, que se encuentra principalmente en la mitocondria y está involucrada en el metabolismo de ácidos grasos de cadena larga. Mutaciones en el gen ECHDC2 se han relacionado con algunos trastornos metabólicos, como la deficiencia de acil-CoA deshidrogenasa de cadena muy larga, aunque se necesita más investigación para comprender mejor la función biológica de ECHDC2 y cómo se relaciona con la fisiología y la patología humanas. Si bien la relación entre ECHDC2 y el cáncer aún no está completamente comprendida, se ha encontrado que la expresión de ECHDC2 está alterada en varios tipos de tumores, incluyendo el cáncer de pulmón y el cáncer colorrectal.")
        elif selected_dataset == 'EGFR':
            st.write("Es una proteína transmembrana que se encuentra en la superficie de muchas células en el cuerpo humano, y es un receptor que se une a las moléculas de señalización llamadas factores de crecimiento epidérmico (EGF) y proteínas relacionadas.Se ha demostrado que EGFR está involucrado en la patogénesis de algunos tipos de cáncer, incluyendo el cáncer de pulmón, el cáncer colorrectal y el cáncer de mama, y se han desarrollado inhibidores de EGFR como tratamientos para estos tipos de cáncer.")
        elif selected_dataset == 'CALM1':
            st.write("Es un gen humano que codifica la proteína calmodulina-1, que es una proteína importante en la señalización celular y la regulación del calcio en las células. La proteína CALM1 también ha sido implicada en la patogénesis de algunos trastornos neurológicos, incluyendo la enfermedad de Alzheimer y la esquizofrenia. La calmodulina ha sido implicada en la tumorigénesis y en la progresión del cáncer en varios tipos de tumores, incluyendo el cáncer de mama, el cáncer de próstata y el cáncer colorrectal.")
        elif selected_dataset == 'APOD':
            st.write("APOD es el acrónimo del gen humano apolipoproteína D, que codifica una proteína que se encuentra en el plasma sanguíneo y en muchos tejidos del cuerpo humano, incluyendo el cerebro, el hígado y el riñón. APOD también ha sido implicada en la protección neuronal contra el daño oxidativo y en la regulación de la inflamación y la respuesta inmune. APOD se ha relacionado con algunos trastornos neurológicos, como la enfermedad de Alzheimer y la esclerosis múltiple, y se ha sugerido que podría tener un papel en la patogénesis de algunos tipos de cáncer.")
        elif selected_dataset == 'SPOCK1':
            st.write("SPOCK1 es el acrónimo del gen humano SPARC/osteonectina, proteína codificada por la cDNA, y kringle conteniendo 1, que codifica la proteína SPOCK1. La proteína SPOCK1 ha sido implicada en la progresión tumoral y la invasión metastásica en algunos tipos de cáncer, incluyendo el cáncer de mama, el cáncer de pulmón y el cáncer de ovario.")
        elif selected_dataset == 'ANXA1':
            st.write("ANXA1 es el acrónimo del gen humano anexina A1, que codifica una proteína llamada anexina A1. ANXA1 se ha relacionado con una variedad de trastornos, incluyendo la inflamación crónica, la artritis reumatoide, la enfermedad inflamatoria intestinal y algunos tipos de cáncer.")
        elif selected_dataset == 'PTGDS':
            st.write("PTGDS es el acrónimo del gen humano prostaglandina D2 sintasa, que codifica una enzima llamada prostaglandina D2 sintasa. Se ha relacionado la enzima PTGDS con una variedad de trastornos, incluyendo la inflamación, la aterosclerosis, la enfermedad de Alzheimer y algunos tipos de cáncer.")
        elif selected_dataset == 'EIF1':
            st.write("EIF1 es el acrónimo del gen humano factor de iniciación 1 de la traducción, que codifica una proteína llamada factor de iniciación eucariota 1 (eIF1). Se ha demostrado que la alteración de la expresión o la función de eIF1 puede tener efectos significativos sobre la síntesis de proteínas y el desarrollo celular, y se ha relacionado con una variedad de trastornos, incluyendo el cáncer, las enfermedades neurodegenerativas y las enfermedades cardiovasculares.")
        elif selected_dataset == 'VIM':
            st.write("VIM es el acrónimo del gen humano vimentina, que codifica una proteína estructural llamada vimentina. Se ha relacionado la vimentina con una variedad de trastornos, incluyendo el cáncer, las enfermedades cardiovasculares y las enfermedades neurodegenerativas.")
        elif selected_dataset == 'MGLL':
            st.write("MGLL es el acrónimo del gen humano monoacilglicerol lipasa, que codifica una enzima llamada monoacilglicerol lipasa. La alteración de la expresión o la función de MGLL se ha relacionado con una variedad de trastornos, incluyendo la obesidad, la diabetes y las enfermedades cardiovasculares, al parecer no tiene relación con el cáncer.")
        elif selected_dataset == 'ITM2C':
            st.write("ITM2C es el acrónimo del gen humano proteína 2 transmembrana del tipo 2C, que codifica una proteína transmembrana de tipo II llamada E25B. La proteína ITM2C se expresa en muchos tejidos diferentes del cuerpo, incluyendo el cerebro, el tejido adiposo y el tejido óseo, y se ha relacionado con una variedad de trastornos, incluyendo el cáncer, las enfermedades neurodegenerativas y la osteoporosis.")
        elif selected_dataset == 'PLLP':
            st.write("PLLP es el acrónimo del gen humano proteína relacionada con la proteolipina, que codifica una proteína llamada proteína relacionada con la proteolipina (PLP). La proteína PLP se expresa principalmente en el cerebro y la médula espinal, y las mutaciones en el gen que codifica PLP se han relacionado con una variedad de trastornos neurológicos, incluyendo la leucodistrofia de Pelizaeus-Merzbacher, una enfermedad genética rara que afecta a la mielina del sistema nervioso central.")
        elif selected_dataset == 'ITGB8':
            st.write("ITGB8 es el acrónimo del gen humano integrina beta 8, que codifica una proteína llamada integrina beta 8. Las mutaciones en el gen que codifica ITGB8 se han relacionado con una variedad de trastornos, incluyendo la enfermedad pulmonar intersticial y la enfermedad cerebrovascular. Además, se ha demostrado que la expresión anormal de ITGB8 está implicada en el desarrollo de varios tipos de cáncer.")
        elif selected_dataset == 'HES6':
            st.write("HES6 es el acrónimo del gen humano factor de regulación de la expresión de la proteína de la piel 6, que codifica una proteína llamada Hairy and Enhancer of Split 6 (HES6). Se ha demostrado que la expresión anormal de HES6 está implicada en el desarrollo de varios tipos de cáncer, incluyendo el cáncer de mama y el cáncer de próstata. Además, se ha relacionado con trastornos neurológicos como la enfermedad de Alzheimer y la esquizofrenia.")
        elif selected_dataset == 'RPS27L':
            st.write("RPS27L es el acrónimo del gen humano proteína ribosomal S27 como símil, que codifica una proteína llamada ribosomal protein S27-like (RPS27L). Se ha demostrado que la expresión anormal de RPS27L está relacionada con varios tipos de cáncer, incluyendo el cáncer de mama, el cáncer de próstata y el cáncer de pulmón. Además, se ha relacionado con trastornos neurológicos como la enfermedad de Parkinson.")
        elif selected_dataset == 'GFAP':
            st.write("GFAP es el acrónimo del gen humano proteína ácida fibrilar glial, que codifica una proteína llamada proteína ácida fibrilar glial (GFAP). La expresión anormal de GFAP se ha relacionado con varios trastornos neurológicos, incluyendo la enfermedad de Alzheimer, la esclerosis múltiple y la epilepsia. No se ha encontrado relación con el cáncer")
        elif selected_dataset == 'TRIM2':
            st.write("TRIM2 es el acrónimo del gen humano proteína 2 con dedos de tipo RING y motivos de leucina, que codifica una proteína llamada TRIM2 (también conocida como tripartite motif-containing protein 2). Se ha demostrado que la proteína TRIM2 desempeña un papel importante en la regulación de la morfología y función de las neuronas, incluyendo la formación de dendritas y la maduración sináptica. La expresión anormal de TRIM2 se ha relacionado con trastornos neurológicos como la esquizofrenia y la epilepsia. No se ha encontrado relación con el cáncer")
        elif selected_dataset == 'APOE':
            st.write("APOE es el acrónimo del gen humano apolipoproteína E, que codifica una proteína llamada apolipoproteína E. Además, también se ha demostrado que APOE ε4 se asocia con un mayor riesgo de enfermedad cardiovascular y de accidente cerebrovascular. No se ha encontrado relación con el cáncer")
        elif selected_dataset == 'ANXA5':
            st.write("ANXA5 es el acrónimo del gen humano anexina A5, que codifica una proteína llamada anexina A5 (también conocida como anexina V). Además, ANXA5 también se ha relacionado con la apoptosis celular y la inflamación. Se ha demostrado que la deficiencia de ANXA5 se asocia con un mayor riesgo de trombosis y abortos espontáneos recurrentes en mujeres embarazadas. No se ha encontrado relación con el cáncer")
        elif selected_dataset == 'NAV1':
            st.write("NAV1 es una abreviación comúnmente utilizada para referirse al canal de sodio dependiente de voltaje tipo 1, que está codificado por el gen SCN1A en los humanos. Las mutaciones en el gen SCN1A se han relacionado con varias enfermedades neurológicas, incluyendo la epilepsia y el trastorno del espectro autista. No se ha encontrado relación con el cáncer")
        elif selected_dataset == 'TMSB4X':
            st.write("TMSB4X es el nombre del gen humano que codifica la proteína tímica de precursores de timosina beta-4 (TMSB4), también conocida como la proteína de la familia de la timosina beta-4 (TB4). Se ha relacionado con la protección del corazón y el cerebro contra el daño isquémico, y se ha investigado como un posible tratamiento para las enfermedades cardiovasculares y neurológicas. En los humanos, la proteína TMSB4X se expresa principalmente en el timo, pero también se encuentra en otros tejidos como la piel, los músculos y el cerebro. No se ha relacionado con el cáncer")
        elif selected_dataset == 'HSPB1':
            st.write("HSPB1 es el nombre de un gen humano que codifica la proteína de choque térmico 27 (HSP27), también conocida como proteína relacionada con la proteína quinasa mitógena 1 (PRKM1P1).  Además, la proteína HSP27 también se ha relacionado con la regulación de la apoptosis (muerte celular programada) y la modulación de la respuesta inmunitaria. Las mutaciones en el gen HSPB1 se han relacionado con varias enfermedades humanas, incluyendo distrofia muscular y neuropatías.")
        elif selected_dataset == 'SEC61G':
            st.write("Es un gen que codifica una subunidad de la proteína translocon, que es un complejo proteico en la membrana del retículo endoplásmico (ER) que está involucrado en la translocación de proteínas a través de la membrana del ER durante su síntesis. Se ha encontrado que la expresión anormal de SEC61G se asocia con varios tipos de cáncer, como el cáncer de pulmón y el cáncer de mama. Además, SEC61G está involucrado en la regulación de la señalización de la proteína Wnt, que está desregulada en muchos tipos de cáncer.")
        elif selected_dataset == 'IGSF6':
            st.write("Es un gen que codifica una proteína de la superfamilia de inmunoglobulinas, que se expresa principalmente en las células del sistema nervioso central. Aunque no se ha demostrado una relación directa entre la mutación en IGSF6 y el cáncer, se ha informado que la expresión anormal de IGSF6 se asocia con la progresión del cáncer de próstata y la resistencia a la terapia hormonal.")
        elif selected_dataset == 'IGFBP2':
            st.write("es un gen que codifica la proteína unida al factor de crecimiento similar a la insulina 2 (IGFBP2), que se une y regula los efectos biológicos del factor de crecimiento similar a la insulina 1 (IGF1) y el factor de crecimiento similar a la insulina 2 (IGF2). La expresión anormal de IGFBP2 se ha relacionado con varios tipos de cáncer, incluyendo cáncer de próstata, cáncer de mama, cáncer de ovario y cáncer colorrectal. ")
        elif selected_dataset == 'RPLP1':
            st.write("Es un gen que codifica una proteína ribosomal, que forma parte de la subunidad grande del ribosoma. Además, se ha encontrado que RPLP1 juega un papel importante en la regulación de la apoptosis (muerte celular programada) y la proliferación celular en células cancerosas. ")
        elif selected_dataset == 'CSF1R':
            st.write("Es un gen que codifica el receptor del factor estimulador de colonias de macrófagos 1 (CSF1), que se expresa en células del sistema inmunológico y juega un papel importante en la diferenciación, supervivencia y función de los macrófagos.  Se ha encontrado que la sobreexpresión de CSF1R se relaciona con varios tipos de cáncer, incluyendo cáncer de mama, cáncer de ovario, cáncer de próstata y cáncer colorrectal.")
        elif selected_dataset == 'NACA':
            st.write("Gen que codifica una proteína de unión a calcio y ácidos nucleicos que se encuentra en el núcleo celular y en la mitocondria. i bien no se ha demostrado que NACA esté directamente involucrado en el cáncer, se ha encontrado que NACA se expresa diferencialmente en diferentes tipos de cáncer, como cáncer de pulmón y cáncer de próstata.")
        elif selected_dataset == 'HTRA1':
            st.write("El gen HTRA1 (High-Temperature Requirement A1) codifica para una proteína que pertenece a una familia de proteasas serina. Las mutaciones en el gen HTRA1 se han relacionado con varios tipos de cáncer, incluyendo cáncer de pulmón, cáncer colorrectal, cáncer de páncreas y cáncer de mama. ")
        elif selected_dataset == 'CSF3R':
            st.write("Codifica para un receptor de la proteína de factor estimulante de colonias 3 (CSF3), también conocida como factor estimulante de colonias de granulocitos (G-CSF). Las mutaciones en el gen CSF3R se han relacionado con varias enfermedades malignas de la sangre, como la leucemia mieloide crónica atípica y la neutropenia congénita severa. ")
        elif selected_dataset == 'CREG1':
            st.write("El gen CREG1 (Cellular Repressor of E1A-stimulated Genes 1) codifica para una proteína que se ha encontrado que regula la expresión de otros genes en diferentes tipos de células, incluyendo células musculares, células nerviosas y células inmunitarias. Las mutaciones en el gen CREG1 se han relacionado con varios tipos de cáncer, incluyendo cáncer de mama, cáncer de próstata, cáncer de páncreas y cáncer colorrectal. ")
        elif selected_dataset == 'FAM107B':
            st.write("El gen FAM107B (Family with sequence similarity 107 member B) codifica para una proteína que se ha encontrado que juega un papel importante en la proliferación celular y la supervivencia de las células. Las mutaciones en el gen FAM107B se han relacionado con varios tipos de cáncer, incluyendo cáncer de pulmón, cáncer de mama y cáncer de ovario.")
        elif selected_dataset == 'SLAMF9':
            st.write("El gen SLAMF9 (Signaling Lymphocytic Activation Molecule Family Member 9) codifica para una proteína que se ha encontrado que juega un papel en la activación y regulación de las células del sistema inmunológico. Las mutaciones en el gen SLAMF9 no se han relacionado con un mayor riesgo de desarrollar cáncer. Sin embargo, se ha encontrado que la proteína SLAMF9 puede estar implicada en la respuesta inmune contra el cáncer.")
        elif selected_dataset == 'GLDN':
            st.write("El gen GLDN (Gliomedin) codifica para una proteína que se encuentra principalmente en las células de Schwann, que son células de soporte que rodean y aíslan las fibras nerviosas periféricas en el sistema nervioso. Las mutaciones en el gen GLDN no se han relacionado con un mayor riesgo de desarrollar cáncer.")
        elif selected_dataset == 'EMP3':
            st.write("El gen EMP3 (Epithelial Membrane Protein 3) codifica para una proteína que se encuentra en la membrana celular de varias células, incluyendo células epiteliales, células nerviosas y células del sistema inmunológico. Las mutaciones en el gen EMP3 se han relacionado con varios tipos de cáncer, incluyendo cáncer de mama, cáncer de pulmón y cáncer de páncreas.")
        elif selected_dataset == 'COMMD6':
            st.write("Codifica para una proteína que se ha encontrado que está involucrada en la regulación del metabolismo del cobre y la homeostasis del cobre en las células. Las mutaciones en el gen COMMD6 no se han relacionado con un mayor riesgo de desarrollar cáncer. ")
        elif selected_dataset == 'ANXA2':
            st.write("El gen ANXA2 (Annexin A2) codifica para una proteína que se encuentra en la membrana celular y en el citoplasma de muchas células diferentes. La proteína ANXA2 está involucrada en una variedad de procesos celulares, incluyendo la adhesión celular, la migración y la endocitosis. Las mutaciones en el gen ANXA2 no se han relacionado con un mayor riesgo de desarrollar cáncer.")
        elif selected_dataset == 'RPL38':
            st.write("El gen RPL38 (Ribosomal Protein L38) codifica para una proteína ribosomal que se encuentra en el complejo ribosomal, el cual está involucrado en la síntesis de proteínas a partir de ARNm en las células. Las mutaciones en el gen RPL38 no se han relacionado con un mayor riesgo de desarrollar cáncer.")
        elif selected_dataset == 'CEBPD':
            st.write("El gen CEBPD (CCAAT/enhancer binding protein delta) codifica para un factor de transcripción que regula la expresión de otros genes en la célula. La proteína CEBPD es importante en la regulación de la diferenciación celular, la inflamación y la respuesta al estrés. Las mutaciones en el gen CEBPD no se han relacionado con un mayor riesgo de desarrollar cáncer.")
        elif selected_dataset == 'APBB1IP':
            st.write("El gen APBB1IP (Amyloid Beta Precursor Protein Binding Family B Member 1 Interacting Protein) codifica para una proteína que interactúa con la proteína precursora de la amiloide beta (APP), que está involucrada en la formación de placas amiloides en el cerebro, una característica distintiva de la enfermedad de Alzheimer. Las mutaciones en el gen APBB1IP no se han relacionado con un mayor riesgo de desarrollar cáncer. Sin embargo, se ha demostrado que la proteína APBB1IP está involucrada en la regulación de la apoptosis, la respuesta al estrés y la dinámica de los microtúbulos.")
        elif selected_dataset == 'HLADRB6':
            st.write("El gen HLADRB6 es parte del complejo de histocompatibilidad principal (MHC, por sus siglas en inglés) de clase II y codifica para una subunidad beta de la molécula de antígeno del MHC de clase II. No se ha informado que las mutaciones en HLADRB6 estén directamente relacionadas con un mayor riesgo de desarrollar cáncer.")
        elif selected_dataset == 'TUBGCP2':
            st.write("El gen TUBGCP2 (Tubulin Gamma Complex Associated Protein 2) codifica para una proteína que forma parte de un complejo proteico conocido como gamma-tubulina complejo, que es esencial para la nucleación y organización de microtúbulos en la célula. Mutaciones en TUBGCP2 no se han relacionado directamente con un mayor riesgo de desarrollar cáncer. Sin embargo, se ha demostrado que la sobreexpresión de TUBGCP2 está relacionada con la proliferación celular aumentada y la progresión tumoral en varios tipos de cáncer, incluyendo el cáncer de ovario, el cáncer de mama y el cáncer colorrectal.")
        elif selected_dataset == 'LCP2':
            st.write("El gen LCP2 (también conocido como SLP-76) codifica para una proteína adaptadora que se asocia con los receptores de células T y juega un papel importante en la transducción de señales en la célula T. Mutaciones en el gen LCP2 no se han relacionado directamente con un mayor riesgo de desarrollar cáncer. Sin embargo, se ha demostrado que la sobreexpresión de LCP2 está relacionada con la progresión tumoral y la metástasis en varios tipos de cáncer, incluyendo el cáncer de próstata, el cáncer de mama y el cáncer de colon.")
        elif selected_dataset == 'LOC100505854':
            st.write("LOC100505854 es un identificador genómico que se asigna a una región del genoma humano que aún no se ha caracterizado completamente y se conoce como una región no codificante del genoma (también llamada ADN basura o ADN no codificante). Actualmente no se conocen mutaciones específicas en LOC100505854 que estén directamente relacionadas con el cáncer.")
        elif selected_dataset == 'IFI44':
            st.write("El gen IFI44 (Interferon-induced protein 44) codifica para una proteína que se expresa en respuesta a la estimulación por interferones tipo I y tipo II, que son proteínas importantes en la respuesta inmune antiviral y antitumoral. Se ha sugerido que IFI44 puede tener un papel en la promoción del crecimiento y la supervivencia de las células tumorales a través de su capacidad para regular la respuesta inmunitaria y la inflamación. Además, se ha encontrado que la sobreexpresión de IFI44 puede estar relacionada con la progresión del cáncer y la resistencia a la quimioterapia.")
        elif selected_dataset == 'GNG11':
            st.write("El gen GNG11 codifica para una subunidad gamma de proteína G, que forma parte de la familia de proteínas G, que son importantes en la señalización celular. se ha encontrado que la expresión de GNG11 está aumentada en varios tipos de cáncer, incluyendo el cáncer de mama, el cáncer de próstata y el cáncer de ovario.")

    with col3:
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

    st.subheader("Análisis de PCA en 3D")
    # Contenido de la página principal
    st.write('Ver el análisis de PCA para una cantidad determinada de genes relevantes en el Glioblastoma.')

    # Creamos la estructura de los datos

    # Creamos una lista con los nombres de los conjuntos de datos y una lista con las cantidades correspondientes
    PCA_images = {'12 Genes':'./images/pca12gen.png', '8 Genes': './images/pca8gen.png'}

    # Creamos un desplegable para que el usuario pueda seleccionar el conjunto de datos a graficar
    selected_dataset = st.selectbox('Selecciona la cantidad de genes a los cuales desea aplicar una visualizacion en 3D usando PCA:', list(PCA_images.keys()))

    # Cargar la imagen correspondiente a la opción seleccionada
    image = Image.open(PCA_images[selected_dataset])
    st.image(image, caption='Análisis de Componentes Principales en 3D')
        