import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagenes(imagenes, titulos, filas=2, cols=3):
    """Función auxiliar para visualización adaptativa"""
    total = len(imagenes)
    fig, axes = plt.subplots(filas, cols, figsize=(16, 10))
    axes = axes.ravel()
    
    for i in range(len(axes)):
        if i < total:
            img = imagenes[i]
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(titulos[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def obtener_kernel_dinamico(shape, ratio):
    """Calcula un kernel impar basado en el tamaño de la imagen"""
    dim_min = min(shape[:2])
    k = int(dim_min * ratio)
    if k % 2 == 0: k += 1
    return (k, k)

def segmentar_monedas_final(ruta_imagen):
    print("="*60)
    print("PROCESAMIENTO ROBUSTO DE MONEDAS Y DADOS (V4)")
    print("="*60)
    
    # 1. CARGA DE IMAGEN
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print("Error: No se pudo cargar la imagen.")
        return

    h, w = img_original.shape[:2]
    area_total = h * w
    print(f"Dimensiones: {w}x{h} píxeles")

    # 2. PREPROCESAMIENTO Y CLAHE
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: Mejora el contraste local (clave para monedas oscuras)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    # 3. ESTIMACIÓN DE FONDO (CORREGIDO)
    # Aumentamos drásticamente el ratio a 0.12 (12% de la imagen).
    # En 3000px, el kernel será ~360px. Esto asegura que el fondo 
    # ignore completamente las monedas.
    k_blur = obtener_kernel_dinamico((h, w), 0.12)
    img_fondo = cv2.GaussianBlur(img_clahe, k_blur, 0)
    
    print(f"Kernel de Fondo calculado: {k_blur} (Suficiente para rellenar objetos)")

    # 4. EXTRACCIÓN DE OBJETOS (AbsDiff)
    # Usamos diferencia absoluta para capturar lo claro (dados) y lo oscuro (monedas cobre)
    img_diff = cv2.absdiff(img_clahe, img_fondo)
    
    # Normalización para estirar el histograma
    img_diff = cv2.normalize(img_diff, None, 0, 255, cv2.NORM_MINMAX)

    # 5. UMBRALIZACIÓN
    _, img_thresh = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. MORFOLOGÍA (LIMPIEZA)
    # Kernel de apertura pequeño para ruido
    k_apertura = obtener_kernel_dinamico((h, w), 0.002)
    # Kernel de cierre mediano para solidificar objetos
    k_cierre = obtener_kernel_dinamico((h, w), 0.005)
    
    img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, k_apertura, iterations=1)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, k_cierre, iterations=3)

    # 7. DETECCIÓN Y FILTRADO
    contours, _ = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_validos = []
    img_resultados = img_original.copy()
    
    # Ajuste de umbrales de área (más permisivos)
    # Min: 0.05% de la imagen (para capturar monedas de 10 centavos)
    min_area = area_total * 0.0005
    max_area = area_total * 0.20
    
    print(f"Filtro de Área: {int(min_area)} - {int(max_area)} px")
    
    # Contadores para depuración
    descartados_area = 0
    descartados_forma = 0
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0: continue
            
            # Circularidad (Relajamos el filtro a 0.2 para aceptar dados deformados)
            circularidad = 4 * np.pi * (area / (perimetro * perimetro))
            
            if circularidad > 0.2:
                contornos_validos.append(cnt)
                
                # Visualización
                cv2.drawContours(img_resultados, [cnt], -1, (0, 255, 0), 3)
                
                # Bounding box y etiqueta
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                # Etiqueta con fondo para legibilidad
                cv2.rectangle(img_resultados, (x, y-25), (x+60, y), (0, 255, 0), -1)
                cv2.putText(img_resultados, f"#{len(contornos_validos)}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                descartados_forma += 1
        else:
            descartados_area += 1

    print(f"\nRESULTADOS:")
    print(f"  - Objetos Detectados: {len(contornos_validos)}")
    print(f"  - Descartados por área (ruido/artefactos): {descartados_area}")
    print(f"  - Descartados por forma (líneas/manchas): {descartados_forma}")

    # 8. VISUALIZACIÓN FINAL
    imagenes = [img_clahe, img_fondo, img_diff, img_thresh, img_clean, img_resultados]
    titulos = ['1. Entrada + CLAHE', '2. Estimación Fondo', '3. Diferencia (AbsDiff)', 
               '4. Umbral (Otsu)', '5. Morfología', f'6. Resultado ({len(contornos_validos)} objs)']
    
    mostrar_imagenes(imagenes, titulos, filas=2, cols=3)


segmentar_monedas_final('Monedas.jpg')