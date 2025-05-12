# TrafficSpeedCam

**TrafficSpeedCam** es un sistema de análisis de vídeo diseñado para cámaras de tráfico. Utiliza los algoritmos **YOLO** (You Only Look Once) para la detección de objetos y **SORT** (Simple Online and Realtime Tracking) para el seguimiento de vehículos, permitiendo calcular la **velocidad del tráfico** en tiempo real.

![frame_00178](https://github.com/user-attachments/assets/9c862262-9ddf-4d9f-9d0c-6aeb8cea52be)


## Características

- Detección precisa de vehículos usando YOLOv11 (https://github.com/ultralytics)
- Seguimiento en tiempo real con SORT (https://github.com/abewley/sort)
- Visualización en vídeo con bounding boxes y velocidad estimada.
- Posibilidad de definir zonas o líneas de control para medir velocidad.

## 🚀 Requisitos

- Python 3.8 o superior
- OpenCV
- NumPy
- YOLOv11
- SORT (archivo local)

Instalación de dependencias:

```bash
pip install -r requirements.txt
