import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from utiles import save_object

# Cargar datos
df = pd.read_csv('DS_Prediccion_de_Cardiopatia_SinDatosPerdidos.csv', sep=';')

# Separar variables independientes y dependiente
X = df.drop('Resultado', axis=1).values
y = df['Resultado'].values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el escalador
save_object('scaler.pkl', scaler)

# Separar conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=30)

# Crear modelo MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.3)

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')
