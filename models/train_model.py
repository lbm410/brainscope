import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from models.cnn_model import crear_modelo_cnn


def entrenar_modelo(data_dir, output_model_path, input_shape=(128, 128, 1), num_clases=4, epochs=10, batch_size=32):
    """
    Entrena el modelo CNN con las imágenes procesadas.
    :param data_dir: Directorio donde se encuentran las imágenes clasificadas en carpetas por tipo.
    :param output_model_path: Ruta para guardar el modelo entrenado.
    :param input_shape: Dimensiones de entrada de las imágenes.
    :param num_clases: Número de clases para la clasificación.
    :param epochs: Número de épocas para el entrenamiento.
    :param batch_size: Tamaño del batch.
    """
    print("Preparando datos para entrenamiento...")

    # Generador de datos con aumentación
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )

    # Crear modelo
    modelo = crear_modelo_cnn(input_shape=input_shape, num_clases=num_clases)

    # Entrenar el modelo
    print("Entrenando modelo...")
    modelo.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Guardar el modelo entrenado
    modelo.save(output_model_path)
    print(f"Modelo guardado en {output_model_path}")
