import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pycaret.regression import load_model, predict_model
import joblib
import os
import requests
from io import BytesIO


#==================================================================================
# URL del archivo .pkl en el repositorio de GitHub
url = 'https://github.com/vmartinezmar/aguamarina/raw/main/house_madrid.pkl'

# Realizar la solicitud HTTP para obtener el archivo .pkl, y añade un tiempo de espera
response = requests.get(url, timeout=10)

# Asegurarse de que la solicitud es exitosa
if response.status_code == 200:
    # Cargar el modelo con joblib del archivo descargado
    model = joblib.load(BytesIO(response.content))
else:
    print('Error al descargar el archivo:', response.status_code)


#==================================================================================
# Cargar el DataFrame entrenado para obtener los valores únicos de las variables categóricas
archivo_kaggle = 'https://raw.githubusercontent.com/vmartinezmar/dataset/main/house_price_madrid_14_08_2022_entrenado.csv'
df = pd.read_csv(archivo_kaggle, encoding='utf-8', sep = ',')

# Obtener los valores únicos para las listas desplegables
house_type_values = np.sort(df['house_type'].unique())
house_type_2_values = np.sort(df['house_type_2'].unique())
district_values = np.sort(df['district'].unique())


#==================================================================================
# Crear el título de la app
st.title('Predicción del Precio de la Vivienda en Madrid, en octubre de 2022.')

# Crear widgets para la entrada de datos
house_type = st.selectbox('Tipo de Casa', house_type_values)
house_type_2 = st.selectbox('Tipo de Casa 2', house_type_2_values)
rooms = st.number_input('Número de Habitaciones', min_value=1, step=1)
m2 = st.number_input('Metros Cuadrados', min_value=25.0, step=1.0)
elevator = st.checkbox('Ascensor')
garage = st.checkbox('Garaje')

district = st.selectbox('Distrito', district_values)

# Filtrar barrios según el distrito seleccionado
neighborhood_values = df[df['district'] == district]['neighborhood'].unique()
neighborhood = st.selectbox('Barrio', neighborhood_values)


#==================================================================================
# Función para formatear los ejes en euros
def euros(x, pos):
    return f'{x:,.2f}€'


#==================================================================================
# Botón para predecir
if st.button('Predecir'):
    with st.spinner('Realizando predicción...'):
        # Crear un DataFrame con los valores de entrada
        input_data = pd.DataFrame({
            'house_type': [house_type],
            'house_type_2': [house_type_2],
            'rooms': [rooms],
            'm2': [m2],
            'elevator': [elevator],
            'garage': [garage],
            'neighborhood': [neighborhood],
            'district': [district]
        })
    
        # Realizar la predicción
        #prediction = predict_model(model, data=input_data)
        prediction = model.predict(input_data)

        # Extraer la predicción
        predicted_price = prediction[0]

        # Mostrar el resultado
        st.write(f'**El precio estimado de la vivienda es: ```{predicted_price:,.2f} €```**')

        # Muestra la predicción completa
        st.write(prediction)


#==================================================================================
        # Filtrar el DataFrame original
        filtered_df = df[#(df['house_type'] == house_type) &
                        #(df['house_type_2'] == house_type_2) &
                        (df['rooms'] == rooms) &
                        #(df['m2'] == m2) &
                        #(df['elevator'] == elevator) &
                        #(df['garage'] == garage) &
                        (df['neighborhood'] == neighborhood) &
                        (df['district'] == district)]

        # Mostrar el DataFrame filtrado
        st.write('**Registros que cumplen con las condiciones de: Distrito, Barrio y Habitaciones.**')
        st.write(filtered_df)


        # Gráfico de nube de puntos
        st.write('**Gráfico de nube de puntos de los registros que cumplen las condiciones de: Distrito, Barrio y Habitaciones.**')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(filtered_df['price'], filtered_df['error'])
        ax.set_title('Nube de puntos')
        ax.set_xlabel('Precio (€)')
        ax.set_ylabel('Error (€)')
        ax.xaxis.set_major_formatter(FuncFormatter(euros))
        ax.yaxis.set_major_formatter(FuncFormatter(euros))
        st.pyplot(fig)


        # Gráfico de residuos similar al de PyCaret
        st.write('**Gráfico de Residuos de todo el dataframe**')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.residplot(x=df['prediction_label'], y=df['error'], lowess=True, ax=ax, line_kws={'color': 'red', 'lw': 1})
        ax.set_title('Residuals Plot')
        ax.set_xlabel('Predicted Price (€)')
        ax.set_ylabel('Residuals (€)')
        ax.xaxis.set_major_formatter(FuncFormatter(euros))
        ax.yaxis.set_major_formatter(FuncFormatter(euros))
        st.pyplot(fig)


        # Mostrar la predicción realizada en el gráfico
        st.write('**Predicción Realizada**')
        st.write(f'**La predicción realizada es: ```{predicted_price:,.2f} €```**')

        # Botón para una nueva predicción
        if st.button('Nueva predicción'):
            st.experimental_rerun()


# Correr la app usando streamlit run app.py