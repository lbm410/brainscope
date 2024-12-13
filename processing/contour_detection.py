import cv2
import numpy as np

def encontrar_contornos(mascara_binaria):
    """
    Encuentra contornos en una máscara binaria.
    :param mascara_binaria: Matriz binaria (1 para píxeles relevantes, 0 para no relevantes).
    :return: Lista de contornos encontrados, cada uno como una lista de coordenadas.
    """
    print("Encontrando contornos...")

    # Asegurarse de que la máscara es del tipo correcto
    mascara_binaria = np.uint8(mascara_binaria)

    # Encontrar los contornos utilizando OpenCV
    contornos, _ = cv2.findContours(
        mascara_binaria,
        cv2.RETR_EXTERNAL,  # Obtener solo los contornos externos
        cv2.CHAIN_APPROX_SIMPLE  # Reducir el número de puntos en el contorno
    )

    # Convertir los contornos a listas de coordenadas
    lista_contornos = [contorno[:, 0, :].tolist() for contorno in contornos]

    print(f"Se encontraron {len(lista_contornos)} contornos.")
    return lista_contornos
