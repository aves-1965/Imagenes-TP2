import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("Monedas.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1) Filtrado homomórfico para corregir iluminación ---
gray_float = gray.astype(np.float32)
log_img = np.log1p(gray_float)

# Filtro Gaussiano para obtener la iluminación
gauss = cv2.GaussianBlur(log_img, (51, 51), 0)

# Restar iluminación (alta frecuencia = objetos)
homomorphic = log_img - gauss

# Normalizar nuevamente
homomorphic = cv2.normalize(homomorphic, None, 0, 255, cv2.NORM_MINMAX)
homomorphic = homomorphic.astype(np.uint8)

# --- 2) Umbral adaptativo (robusto a iluminación variable) ---
binary = cv2.adaptiveThreshold(
    homomorphic,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35,   # tamaño del bloque
    5     # constante de ajuste
)

# --- 3) Limpieza morfológica ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

# Guardar resultado
cv2.imwrite("binaria_corregida.png", binary_clean)

print("Listo. Imagen binaria mejorada generada.")
