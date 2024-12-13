from processing.classification import clasificar_con_cnn

if __name__ == "__main__":
    ruta_imagen = "C:/Users/kekol/Desktop/UAL/TFG/brainscope/data/test/img.png"  # Ruta de la imagen a clasificar
    modelo_path = "C:/Users/kekol/Desktop/UAL/TFG/brainscope/models/hematoma_model.h5"

    # Llama al método de clasificación
    clase, probabilidad = clasificar_con_cnn(
        ruta_imagen=ruta_imagen,
        modelo_path=modelo_path,
        input_shape=(128, 128, 1),
        clases=["Epidural", "Intracerebral", "Subaracnoidea", "Subdural"]
    )

    print(f"Resultado: {clase} con probabilidad {probabilidad:.2f}")
