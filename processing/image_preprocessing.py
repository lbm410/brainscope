import numpy as np
import cv2

def resaltar_246(imagen):
    """
    Resalta los píxeles con valor 246 y elimina los cinco valores más blancos de la imagen.
    :param imagen: Matriz de la imagen en escala de grises.
    :return: Matriz procesada con los valores resaltados.
    """
    print("Resaltando valores 246 y eliminando los valores más blancos...")

    # Convertir la imagen a escala de grises si no lo está
    if len(imagen.shape) == 3:  # Si tiene 3 canales (RGB)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Encontrar los cinco valores más blancos
    valores_unicos, conteo = np.unique(imagen, return_counts=True)
    valores_mas_blancos = valores_unicos[np.argsort(valores_unicos)[-5:]]

    # Crear una copia de la imagen con los valores resaltados
    resaltado = np.zeros_like(imagen)
    for fila in range(imagen.shape[0]):
        for col in range(imagen.shape[1]):
            valor = imagen[fila, col]
            if valor not in valores_mas_blancos and valor == 246:
                resaltado[fila, col] = valor

    print("Resaltado completado.")
    return resaltado

def eliminar_pixeles_baja_densidad(matriz, ventana, umbral):
    """
    Elimina los píxeles con baja densidad de una matriz de imagen.
    :param matriz: Matriz en escala de grises.
    :param ventana: Tamaño de la ventana para analizar los píxeles vecinos.
    :param umbral: Número mínimo de píxeles relevantes en la ventana para conservar un píxel.
    :return: Matriz filtrada.
    """
    print("Eliminando píxeles con baja densidad...")
    filas, columnas = matriz.shape
    resultado = np.zeros_like(matriz)

    # Iterar sobre cada píxel de la matriz
    for i in range(filas):
        for j in range(columnas):
            if matriz[i, j] == 246:  # Píxel relevante
                conteo = 0

                # Analizar vecinos en la ventana definida
                for k in range(-ventana, ventana + 1):
                    for l in range(-ventana, ventana + 1):
                        ni, nj = i + k, j + l
                        if 0 <= ni < filas and 0 <= nj < columnas and matriz[ni, nj] == 246:
                            conteo += 1

                # Conservar el píxel si cumple con el umbral
                if conteo >= umbral:
                    resultado[i, j] = 246

    print("Eliminación de píxeles con baja densidad completada.")
    return resultado

def aplicar_filtro_mediana(matriz, tamaño_ventana=3):
    """
    Aplica un filtro de mediana a una matriz de imagen para reducir el ruido.
    :param matriz: Matriz en escala de grises.
    :param tamaño_ventana: Tamaño de la ventana del filtro de mediana (debe ser un número impar).
    :return: Matriz filtrada.
    """
    print("Aplicando filtro de mediana...")

    # Validar que el tamaño de la ventana sea impar
    if tamaño_ventana % 2 == 0:
        raise ValueError("El tamaño de la ventana debe ser un número impar.")

    # Aplicar el filtro de mediana usando OpenCV
    matriz_filtrada = cv2.medianBlur(matriz, tamaño_ventana)

    print("Filtro de mediana aplicado.")
    return matriz_filtrada

def crear_mascara_binaria(matriz_resaltada):
    """
    Crea una máscara binaria a partir de una matriz resaltada.
    :param matriz_resaltada: Matriz en escala de grises donde los píxeles relevantes tienen valor 246.
    :return: Matriz binaria (1 para píxeles relevantes, 0 para no relevantes).
    """
    print("Creando máscara binaria...")

    # Convertir la matriz resaltada a una máscara binaria
    mascara_binaria = np.where(matriz_resaltada == 246, 1, 0)

    print("Máscara binaria creada.")
    return mascara_binaria
