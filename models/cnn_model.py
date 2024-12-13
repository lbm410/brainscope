import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def crear_modelo_cnn(input_shape=(128, 128, 1), num_clases=3):
    """
    Crea una red neuronal convolucional para clasificar hematomas.
    :param input_shape: Dimensiones de entrada de las imágenes (alto, ancho, canales).
    :param num_clases: Número de clases para la clasificación.
    :return: Modelo compilado de TensorFlow/Keras.
    """
    print("Creando modelo CNN...")

    modelo = Sequential()

    # Primera capa de convolución y pooling
    modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda capa de convolución y pooling
    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Tercera capa de convolución y pooling
    modelo.add(Conv2D(128, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Aplanar las características extraídas
    modelo.add(Flatten())

    # Capas densas
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))  # Regularización para evitar sobreajuste
    modelo.add(Dense(num_clases, activation='softmax'))  # Salida para clasificación

    # Compilar el modelo
    modelo.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    print("Modelo CNN creado y compilado.")
    return modelo
