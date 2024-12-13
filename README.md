# BrainScope

**BrainScope** es una plataforma avanzada para el análisis automatizado de imágenes de TAC craneal. Combina procesamiento de imágenes, inteligencia artificial y una interfaz web intuitiva para detectar y clasificar hematomas intracraneales.

---

## **Características principales**

- **Procesamiento de Imágenes**: Funciones avanzadas para la detección de contornos y la clasificación de hematomas.
- **Interfaz Web con Flask**: Permite a los usuarios cargar imágenes, procesarlas y visualizar los resultados en tiempo real.
- **Red Neuronal Convolucional**: Modelo entrenado para mejorar la precisión en la clasificación de hematomas.

---

## **Estructura del proyecto**

```plaintext
BrainScope/
├── app/                       # Código principal de la aplicación Flask
├── models/                    # Modelos de Machine Learning
├── processing/                # Procesamiento de imágenes
├── tests/                     # Pruebas del proyecto
├── data/                      # Datos y modelos entrenados
├── LICENSE                    # Licencia del proyecto
├── README.md                  # Descripción del proyecto
├── requirements.txt           # Dependencias del proyecto
├── run.py                     # Script principal para ejecutar Flask
└── config.py                  # Configuración de la aplicación Flask
```

---

## **Requisitos**

### Dependencias
Instala las dependencias necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### Archivos requeridos
- **Imágenes**: Las imágenes de TAC deben subirse a través de la interfaz web o estar disponibles en `data/raw/`.

---

## **Instrucciones de uso**

### 1. Clona el repositorio
```bash
git clone https://github.com/tu_usuario/BrainScope.git
cd BrainScope
```

### 2. Ejecuta la aplicación
```bash
python run.py
```
La aplicación estará disponible en `http://127.0.0.1:5000/`.

### 3. Interfaz web
1. Sube una imagen de TAC.
2. Procesa la imagen para visualizar los contornos y la clasificación del hematoma.
3. Descarga los resultados procesados si es necesario.

---

## **Contribuciones**
Si deseas contribuir al proyecto, por favor abre un issue o realiza un pull request. Las sugerencias y mejoras son bienvenidas.

---

## **Licencia**
Este proyecto está bajo los derechos de autor de **Lucas Barrientos Muñoz**. Por favor, consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## **Contacto**
Para preguntas o colaboraciones, puedes contactar a **Lucas Barrientos Muñoz** a través de los canales especificados en el repositorio.
