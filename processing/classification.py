import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def clasificar_con_cnn(ruta_imagen, modelo_path, input_shape=(128, 128, 1), clases=["Epidural", "Intracerebral", "Subaracnoidea", "Subdural"]):
    """
    Clasifica una imagen usando un modelo de red neuronal convolucional.
    :param ruta_imagen: Ruta de la imagen a clasificar.
    :param modelo_path: Ruta del modelo entrenado (archivo .h5).
    :param input_shape: Dimensiones de entrada del modelo (alto, ancho, canales).
    :param clases: Lista de etiquetas de las clases.
    :return: Clase predicha y la probabilidad asociada.
    """
    print(f"Clasificando la imagen: {ruta_imagen}")

    # Cargar el modelo entrenado
    modelo = tf.keras.models.load_model(modelo_path)
    print("Modelo cargado correctamente.")

    # Cargar la imagen, redimensionarla y convertirla a un array
    imagen = load_img(ruta_imagen, target_size=input_shape[:2], color_mode="grayscale")
    imagen_array = img_to_array(imagen) / 255.0  # Normalización [0, 1]
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch

    # Realizar la predicción
    predicciones = modelo.predict(imagen_array)
    clase_index = np.argmax(predicciones)
    probabilidad = np.max(predicciones)

    # Obtener la clase predicha
    clase_predicha = clases[clase_index]

    print(f"Clasificación: {clase_predicha} (Probabilidad: {probabilidad:.2f})")
    return clase_predicha, probabilidad
