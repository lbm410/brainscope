import os
from models.train_model import entrenar_modelo

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio actual del script
    data_dir = os.path.join(base_dir, "../data/processed")  # Ruta absoluta a 'data/processed'

    entrenar_modelo(
        data_dir=data_dir,
        output_model_path=os.path.join(base_dir, "../models/hematoma_model.h5"),
        input_shape=(128, 128, 1),
        num_clases=4,
        epochs=15,
        batch_size=32
    )
