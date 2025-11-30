import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
# AJUSTE: Rango de relación de aspecto más estricto
MIN_PLATE_RATIO = 2.8 
MAX_PLATE_RATIO = 4.5
MIN_PLATE_AREA = 500
"""

# --- CONSTANTES DE FILTRADO ---
# Detección de Placa (Punto A)
MIN_FP = 0.005
MAX_FP = 0.100
MIN_PLATE_RATIO = 1.6
MAX_PLATE_RATIO = 6.0
MIN_PLATE_AREA = 500
# Nueva constante de filtro de Solidez
MIN_PLATE_SOLIDITY = 0.70 # Exigir que el contorno ocupe al menos el 70% de su Bounding Box

# Parámetros del nuevo filtro de Borde Claro
BORDER_SIZE = 1           # Ancho en píxeles de la zona del borde a analizar.
INTENSITY_THRESHOLD = 80 # Valor de gris mínimo para considerar un píxel como "blanco" (0-255).
MIN_WHITE_RATIO = 0.05     # Porcentaje mínimo de píxeles blancos que debe tener el borde (10%).

# Segmentación de Caracteres (Punto B)
MIN_CHAR_AREA = 50
PLACA_SIZE_NORM = (200, 50) # Tamaño de normalización (W, H)


def check_white_border(img_gray, x, y, w, h, border_size=BORDER_SIZE,
                       intensity_threshold=INTENSITY_THRESHOLD,
                       min_white_ratio=MIN_WHITE_RATIO):
    """
    Verifica si hay un porcentaje mínimo de píxeles 'blancos' en la región del borde
    del contorno detectado (bounding box).
    """
    # Manejar caso en que el contorno sea muy pequeño
    if w <= 2 * border_size or h <= 2 * border_size:
        return False

    # 1. Recortar la región completa de la bounding box
    region = img_gray[y:y+h, x:x+w]

    # 2. Crear una máscara para la ZONA CENTRAL (interior de la placa)
    mask_center = np.zeros_like(region, dtype=np.uint8)
    # Dibujar un rectángulo lleno en el centro para definir la zona a excluir
    cv2.rectangle(mask_center, (border_size, border_size),
                  (w - border_size, h - border_size), 255, -1)

    # 3. Crear una máscara para la ZONA DEL BORDE (lo que no es centro)
    mask_border = cv2.bitwise_not(mask_center)

    # 4. Aislar solo los píxeles de la región del borde
    border_pixels = cv2.bitwise_and(region, region, mask=mask_border)

    # 5. Contar los píxeles claros/blancos en el borde (por encima del umbral)
    white_pixels = np.sum(border_pixels > intensity_threshold)

    # 6. Contar el número total de píxeles en el borde
    total_border_pixels = np.sum(mask_border > 0)

    if total_border_pixels == 0:
        return False

    # 7. Calcular el porcentaje de píxeles blancos en el borde
    white_ratio = white_pixels / total_border_pixels

    # Devolver True si el porcentaje cumple el requisito
    return white_ratio >= min_white_ratio


def detectar_placa(imagen_path):
    """
    Punto A: Detecta automáticamente la placa patente y la segmenta.
    MEJORA: Uso de CLAHE para normalizar el contraste y NUEVO FILTRO DE BORDE.
    """
    print(f"--- 🅰️ Iniciando Detección de Placa para {imagen_path} ---")

    img_bgr = cv2.imread(imagen_path)
    if img_bgr is None:
        if imagen_path.endswith('.png'):
            img_bgr = cv2.imread(imagen_path.replace('.png', '.jpg'))
            imagen_path = imagen_path.replace('.png', '.jpg')
        if img_bgr is None:
            print("Error al cargar la imagen. Revise el nombre del archivo.")
            return None

    img_copy = img_bgr.copy()
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. MEJORA: Aplicar CLAHE para mejorar el contraste local (Filtro Adaptativo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast_enhanced = clahe.apply(img_gray)

    # 2. Suavizado y Detección de Bordes (Canny)
    img_filtered = cv2.bilateralFilter(img_contrast_enhanced, 11, 17, 17)
    edged = cv2.Canny(img_filtered, 30, 200)
    print("Etapa 1: Contraste mejorado con CLAHE y bordes con Canny.")

    # 3. Detección de Contornos
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    placa_segmentada = None
    mejor_contorno = None

    # 4. Filtrado de Contornos para encontrar la Placa
    for cnt in contours:
        area = cv2.contourArea(cnt) # Área
        perimeter = cv2.arcLength(cnt, True) # Perímetro

        if perimeter == 0: continue

        fp = area / (perimeter ** 2) # Factor de Forma

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        # MEJORA: Calcular Solidez
        solidez = area / (w * h)

        # 5. Aplicar Criterios de Filtrado Geométrico
        if (area > MIN_PLATE_AREA and
            MIN_FP < fp < MAX_FP and
            MIN_PLATE_RATIO < aspect_ratio < MAX_PLATE_RATIO and
            solidez > MIN_PLATE_SOLIDITY):  # <-- Nueva condición

            # 6. Verificar Borde Claro (Color/Intensidad)
            if check_white_border(img_gray, x, y, w, h):

                mejor_contorno = cnt
                print(f"  -> Posible Placa detectada. Área: {area:.0f}, Fp: {fp:.4f}, Ratio (W/H): {aspect_ratio:.2f} **(Pasó filtro de borde)**")

                # Segmentar (Recortar) la placa
                placa_segmentada = img_gray[y:y+h, x:x+w]
                cv2.drawContours(img_copy, [mejor_contorno], -1, (0, 255, 0), 3)
                break # Placa encontrada

    # 7. Mostrar Resultado de la Detección
    if placa_segmentada is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title('A. Placa Detectada')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(placa_segmentada, cmap='gray')
        plt.title('Placa Segmentada')
        plt.axis('off')
        plt.show()
        return placa_segmentada
    else:
        print(f"  -> No se encontró una placa que cumpla todos los criterios (geométricos y de borde claro). La segmentación de caracteres será omitida.")
        return None

def segmentar_caracteres(placa_img):
    """
    Punto B: Segmenta los caracteres.
    AJUSTE FINO: Cambio de parámetros de Binarización Adaptativa y Tolerancia estadística.
    """
    if placa_img is None:
        print("--- 🅱️ Segmentación de Caracteres Omitida (Placa no detectada) ---")
        return

    print("\n--- 🅱️ Iniciando Segmentación de Caracteres (AJUSTE FINO) ---")

    # 1. Normalización de la Placa
    placa_norm = cv2.resize(placa_img, PLACA_SIZE_NORM, interpolation=cv2.INTER_AREA)

    # 2. Binarización Robusta: Aumentamos el tamaño del bloque para una mejor binarización
    # El bloque 25 es más amplio, promediando más píxeles para el umbral.
    thresh = cv2.adaptiveThreshold(placa_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 4)

    # 3. Operación Morfológica: Aumentamos las iteraciones para fusionar mejor las letras rotas
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2) # Iteraciones 2

    # 4. Detección de Contornos de Caracteres
    char_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    posibles_caracteres = []
    alturas = []

    # Filtro Previo CLAVE: Usamos 30% de la altura normalizada como mínimo absoluto
    MIN_ALTURA_ABSOLUTA = int(PLACA_SIZE_NORM[1] * 0.3)

    for cnt in char_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filtro de Área y ALTURA MÍNIMA ABSOLUTA (Elimina ruido pequeño)
        if area > MIN_CHAR_AREA and h > MIN_ALTURA_ABSOLUTA:
            posibles_caracteres.append({'x': x, 'y': y, 'w': w, 'h': h, 'imagen': placa_norm[y:y+h, x:x+w]})
            alturas.append(h)

    # 5. Filtrado Estadístico Ajustado
    if not alturas:
        print("  -> No se detectaron caracteres válidos después del filtro de altura/área.")
        return

    altura_promedio = np.mean(alturas)
    desviacion_estandar = np.std(alturas)

    # CORRECCIÓN DE TOLERANCIA: Usamos max(3*sigma, 5)
    # Forzamos una tolerancia mínima de 5 píxeles si sigma es muy pequeña/cero (el punto de fallo).
    tolerancia = max(3 * desviacion_estandar, 5)

    caracteres_segmentados = [
        char_data for char_data in posibles_caracteres
        if altura_promedio - tolerancia < char_data['h'] < altura_promedio + tolerancia
    ]

    print(f"Etapa 2: Altura Promedio: {altura_promedio:.2f}, Tolerancia: {tolerancia:.2f}. Segmentados {len(caracteres_segmentados)} caracteres.")

    # 6. Ordenar los caracteres y Mostrar Resultado
    caracteres_segmentados.sort(key=lambda item: item['x'])

    if caracteres_segmentados:
        fig, axes = plt.subplots(1, len(caracteres_segmentados), figsize=(12, 3))
        fig.suptitle('B. Caracteres Segmentados (Ordenados)')

        if len(caracteres_segmentados) == 1: axes = [axes]

        for i, char_data in enumerate(caracteres_segmentados):
            axes[i].imshow(char_data['imagen'], cmap='gray')
            axes[i].set_title(f'Carácter {i+1}', fontsize=8)
            axes[i].axis('off')
        plt.show()
    else:
        print("  -> No se detectaron caracteres válidos después del filtrado estadístico.")

# --- EJECUCIÓN DEL ALGORITMO con img01.png ---
imagen_de_prueba = 'img06.png'
placa_detectada = detectar_placa(imagen_de_prueba)
segmentar_caracteres(placa_detectada)