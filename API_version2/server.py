from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from utiles import load_object
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo y el escalador al inicio
modelo = load_model('modelo_entrenado.h5')
escalador = load_object('scaler.pkl')

# Ruta para la página principal


@app.route('/')
def index():
    return render_template('index.html', prediccion=None)


# Ruta para manejar el formulario y hacer la predicción


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        datos = {
            'Edad': float(request.form['Edad']),
            'Sexo': float(request.form['Sexo']),
            'Tipo_Dolor_Pecho': float(request.form['Tipo_Dolor_Pecho']),
            'Presion_Arterial': float(request.form['Presion_Arterial']),
            'Colesterol': float(request.form['Colesterol']),
            'Nivel_Azucar_Ayunas': float(request.form['Nivel_Azucar_Ayunas']),
            'Res_Electrocardiografico': float(request.form['Res_Electrocardiografico']),
            'Frecuencia_Cardiaca': float(request.form['Frecuencia_Cardiaca']),
            'Angina_Inducida': float(request.form['Angina_Inducida']),
            'DepresionST': float(request.form['DepresionST']),
            'DepresionST_Pendiente': float(request.form['DepresionST_Pendiente']),
            'NumeroVasos': float(request.form['NumeroVasos']),
            'Prueba_Tálio': float(request.form['Prueba_Tálio'])
        }

        # Convertir los datos del formulario en un DataFrame
        df = pd.DataFrame([datos])

        # Escalar los datos
        X_scaled = escalador.transform(df)

        # Hacer la predicción
        y_pred_prob = modelo.predict(X_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Enviar la predicción de vuelta a la página
        probabilidad = float(y_pred_prob[0][0])  # convertir a float normal
        prediccion = int(y_pred[0][0])  # convertir a 0 o 1 normal

        return render_template('index.html', prediccion=prediccion, probabilidad=probabilidad)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8000, debug=True)
