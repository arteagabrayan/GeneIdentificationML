import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Tres columnas en Streamlit", page_icon=":bar_chart:")

# Columnas
col1, col2, col3 = st.columns(3)

# Columna 1
with col1:
    st.header("Columna 1")
    st.subheader("Gráfico 1")
    st.line_chart({"data": [1, 5, 2, 6, 2, 8]})
    st.subheader("Gráfico 2")
    st.bar_chart({"data": [5, 3, 7, 2, 4]})

# Columna 2
with col2:
    st.header("Columna 2")
    st.subheader("Tabla de datos")
    st.table({"Columna 1": [1, 2, 3], "Columna 2": [4, 5, 6], "Columna 3": [7, 8, 9]})
    st.subheader("Gráfico 3")
    st.area_chart({"data": [3, 6, 1, 7, 4, 9]})

# Columna 3
with col3:
    st.header("Columna 3")
    st.subheader("Información")
    opcion = st.selectbox("Seleccione una opción", ["Opción 1", "Opción 2"])
    if opcion == "Opción 1":
        st.write("Esta es la información de la opción 1")
    else:
        st.write("Esta es la información de la opción 2")
