from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo
with open("C:/Users/keyla/Jupyter/ProyectosG/TP3/ModeloMLP.sav", "rb") as f: #CAMBIAR RUTA
    model = pickle.load(f)

# Cargar el scaler
with open("C:/Users/keyla/Jupyter/ProyectosG/TP3/scaler.sav", "rb") as f: #CAMBIAR RUTA
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Obtener datos del formulario
        Edad = int(request.form['Edad'])
        Sexo = int(request.form['Sexo'])
        Tipo_Dolor_Pecho = int(request.form['Tipo_Dolor_Pecho'])
        Presion_Arterial = int(request.form['Presion_Arterial'])
        Colesterol = int(request.form['Colesterol'])
        Nivel_Azucar_Ayunas = int(request.form['Nivel_Azucar_Ayunas'])
        Res_Electrocardiografico = int(request.form['Res_Electrocardiografico'])
        Frecuencia_Cardiaca = int(request.form['Frecuencia_Cardiaca'])
        Angina_Inducida = int(request.form['Angina_Inducida'])
        DepresionST = float(request.form['DepresionST'])
        DepresionST_Pendiente = int(request.form['DepresionST_Pendiente'])
        NumeroVasos = int(request.form['NumeroVasos'])
        Prueba_Talio = int(request.form['Prueba_Tálio'])

        # Crear array de entrada
        input_data = np.array([[Edad, Sexo, Tipo_Dolor_Pecho, Presion_Arterial, Colesterol,
                                Nivel_Azucar_Ayunas, Res_Electrocardiografico, Frecuencia_Cardiaca,
                                Angina_Inducida, DepresionST, DepresionST_Pendiente,
                                NumeroVasos, Prueba_Talio]])

        # Escalar los datos antes de predecir
        input_scaled = scaler.transform(input_data)

        # Realizar la predicción
        prediccion_mlp = model.predict(input_scaled)

        probabilidad_clase_1 = float(prediccion_mlp[0][0])
        probabilidad_clase_0 = 1 - probabilidad_clase_1
        probabilidades = [probabilidad_clase_0, probabilidad_clase_1]

        prediccion_binaria = int(probabilidad_clase_1 > 0.5)

        prediction_text = f"Predicción para el ejemplo:\n[{prediccion_binaria}]\n"
        prediction_text += f"Probabilidades para las clases (0 y 1):\n{probabilidades}"

        return render_template('index.html', prediction_text=prediction_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
