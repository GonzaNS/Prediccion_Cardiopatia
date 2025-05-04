import json
import requests

# Dirección del servidor Flask (ajusta el puerto si es necesario)
url = 'http://127.0.0.1:8000/predict'

# ✅ MEJORA: Simulación de entrada como lista de pacientes (aunque sea uno)
datos = [
    {
        "Edad": 58,
        "Sexo": 1,
        "Tipo_Dolor_Pecho": 2,
        "Presion_Arterial": 140,
        "Colesterol": 230,
        "Nivel_Azucar_Ayunas": 0,
        "Res_Electrocardiografico": 0,
        "Frecuencia_Cardiaca": 165,
        "Angina_Inducida": 1,
        "DepresionST": 1.0,
        "DepresionST_Pendiente": 1,
        "NumeroVasos": 0,
        "Prueba_Tálio": 2
    }
]

# ✅ CAMBIO: Asegurarse que el `Content-Type` es JSON
headers = {'Content-Type': 'application/json'}

# ✅ CAMBIO: Usar json.dumps para asegurar formato correcto
respuesta = requests.post(url, data=json.dumps(datos[0]), headers=headers)

# Mostrar la respuesta
print('Código de estado:', respuesta.status_code)
print('Respuesta del servidor:', respuesta.json())
