from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#Cargamos el modelo
with open("C:/Users/keyla/Jupyter/Prediccion_Cardiopatia/API_version1/ModeloMLP.sav", "rb") as f: #CAMBIAR RUTA
    model = pickle.load(f)

#Cargamos el scaler
with open("C:/Users/keyla/Jupyter/Prediccion_Cardiopatia/API_version1/scaler.sav", "rb") as f: #CAMBIAR RUTA
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        #Obtenemos datos del formulario
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

        #Creamos array de entrada
        input_data = np.array([[Edad, Sexo, Tipo_Dolor_Pecho, Presion_Arterial, Colesterol,
                                Nivel_Azucar_Ayunas, Res_Electrocardiografico, Frecuencia_Cardiaca,
                                Angina_Inducida, DepresionST, DepresionST_Pendiente,
                                NumeroVasos, Prueba_Talio]])

        #Escalamos los datos antes de predecir
        input_scaled = scaler.transform(input_data)

        #Realizamos la predicción
        prediccion_mlp = model.predict(input_scaled)

        probabilidad_clase_1 = float(prediccion_mlp[0][0])
        prediccion_binaria = int(probabilidad_clase_1 > 0.5)

        if prediccion_binaria == 1:
            resultado = "Positivo"
            probabilidad = round(probabilidad_clase_1 * 100, 2)
        else:
            resultado = "Negativo"
            probabilidad = round((1 - probabilidad_clase_1) * 100, 2)

        prediction_text = f"Resultado del diagnóstico: {resultado} ({probabilidad}%)"

        return render_template('index.html', prediction_text=prediction_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
