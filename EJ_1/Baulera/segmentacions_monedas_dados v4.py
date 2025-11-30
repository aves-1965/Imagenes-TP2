import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagenes(imagenes, titulos, filas=2, cols=3, figsize=(15, 10)):
    """Función auxiliar para mostrar múltiples imágenes"""
    fig, axes = plt.subplots(filas, cols, figsize=figsize)
    axes = axes.ravel()
    
    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        if i < len(axes):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(titulo)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def obtener_kernel_dinamico(shape, ratio):
    """
    Calcula un tamaño de kernel impar basado en una proporción de la imagen.
    Args:
        shape: Tupla con dimensiones (alto, ancho)
        ratio: Porcentaje de la dimensión mínima (0.0 a 1.0)
    Returns:
        Tupla (k, k) con k impar.
    """
    dim_min = min(shape[:2])
    k = int(dim_min * ratio)
    if k % 2 == 0:
        k += 1
    return (k, k)

def segmentar_monedas_mejorado(ruta_imagen):
    print("="*60)
    print("PROCESAMIENTO DE IMAGEN: ENFOQUE DINÁMICO CON CLAHE")
    print("="*60)
    
    # 1. Carga
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print("Error al cargar imagen")
        return
        
    h, w = img_original.shape[:2]
    area_total = h * w
    print(f"Dimensiones: {w}x{h} ({area_total} px)")

    # 2. Preprocesamiento y CLAHE
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # APLICACIÓN DE CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Esto mejora el contraste local para separar monedas oscuras del fondo oscuro
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # 3. Estimación de Fondo Dinámica
    # Ratio 0.02 (2%) suele ser buen tamaño para estimar fondo sin borrar objetos pequeños
    k_blur = obtener_kernel_dinamico(img_original.shape, 0.025)
    img_blur = cv2.GaussianBlur(img_clahe, k_blur, 0)
    
    # 4. Diferencia Absoluta (SOLUCIÓN CLAVE)
    # Usamos absdiff en lugar de subtract. Esto captura objetos que sean
    # más brillantes O más oscuros que el fondo.
    img_diff = cv2.absdiff(img_clahe, img_blur)
    
    # Normalizamos para aprovechar el rango completo 0-255 antes de umbralizar
    img_diff = cv2.normalize(img_diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. Umbralización (Otsu)
    # Al haber eliminado el fondo con absdiff, el histograma es bimodal (ruido vs señal)
    _, img_thresh = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Morfología Matemática Dinámica
    # Kernel pequeño para ruido (0.2% de la imagen)
    k_morph_apertura = obtener_kernel_dinamico(img_original.shape, 0.002) 
    # Kernel mediano para cerrar huecos dentro de monedas/dados (0.6% de la imagen)
    k_morph_cierre = obtener_kernel_dinamico(img_original.shape, 0.006)
    
    # Apertura para quitar ruido de fondo
    img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, k_morph_apertura, iterations=1)
    # Cierre para rellenar los puntos de los dados o texturas de monedas
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, k_morph_cierre, iterations=2) # 2 iters para asegurar cierre
    
    print(f"Kernel Blur: {k_blur}")
    print(f"Kernel Apertura: {k_morph_apertura}")
    print(f"Kernel Cierre: {k_morph_cierre}")

    # 7. Detección y Filtrado Dinámico
    contours, _ = cv2.findContours(img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_validos = []
    img_resultados = img_original.copy()
    
    # Criterios dinámicos de área (evitamos hardcoding de píxeles)
    # Ajustados para detectar desde monedas pequeñas hasta agrupaciones
    area_min_ratio = 0.001  # 0.1% del total de la imagen
    area_max_ratio = 0.05   # 5% del total (para evitar detectar el fondo entero si falla algo)
    
    min_area = area_total * area_min_ratio
    max_area = area_total * area_max_ratio
    
    print(f"Filtro de Área: {int(min_area)} px - {int(max_area)} px")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtro por área
        if min_area < area < max_area:
            # Filtro adicional por circularidad/convexidad para descartar ruido irregular
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0: continue
            circularidad = 4 * np.pi * (area / (perimetro * perimetro))
            
            # Las monedas son circulares (>0.7), los dados cuadrados (>0.5 aprox)
            # Filtramos líneas o ruido muy alargado
            if circularidad > 0.4: 
                contornos_validos.append(cnt)
                
                # Dibujar bounding box y contorno
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_resultados, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Etiqueta
                cv2.putText(img_resultados, f"#{len(contornos_validos)}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"\nTOTAL OBJETOS DETECTADOS: {len(contornos_validos)}")

    # Visualización
    titulos = [
        'Original + CLAHE', 
        'Extracción de Fondo (AbsDiff)', 
        'Umbralización (Otsu)',
        f'Morfología (Cierre K={k_morph_cierre[0]})',
        'Detección Final'
    ]
    
    imagenes = [
        img_clahe,
        img_diff,
        img_thresh,
        img_closing,
        img_resultados
    ]
    
    # Mostramos en disposición personalizada para ver detalle
    mostrar_imagenes(imagenes, titulos, filas=2, cols=3)

# Ejecutar
# if __name__ == "__main__":
    # Asegúrate de que el nombre coincida con tu archivo subido
segmentar_monedas_mejorado('Monedas.jpg')