import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('Monedas.png')
if img is None:
    raise ValueError('No se pudo cargar la imagen.')

# Convertir a gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizado
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Binarización por Otsu
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Guardar resultado
cv2.imwrite('binaria.png', binary)

