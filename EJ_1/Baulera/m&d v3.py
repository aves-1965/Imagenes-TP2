import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Funciones auxiliares (manteniendo la estructura) ---

def mostrar_imagenes(imagenes, titulos, filas=2, cols=3, figsize=(16, 10)):
    total = len(imagenes)
    fig, axes = plt.subplots(filas, cols, figsize=figsize)
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

def obtener_kernel_impar(valor):
    """Asegura que el tamaño del kernel sea impar y entero"""
    k = int(valor)
    if k % 2 == 0:
        k += 1
    return k

def segmentar_monedas_robustez(ruta_imagen):
    print("="*60)
    print("PROCESAMIENTO V6: UMBRAL ADAPTATIVO A GRAN ESCALA")
    print("="*60)
    
    # 1. CARGA
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print("Error: No se pudo cargar la imagen.")
        return

    h, w = img_original.shape[:2]
    dim_min = min(h, w)
    area_total = h * w

    # 2. PREPROCESAMIENTO (GRAY + CLAHE)
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # Suavizado: Necesario, pero manteniéndolo pequeño
    k_blur = obtener_kernel_impar(dim_min * 0.003)
    img_blur = cv2.GaussianBlur(img_clahe, (k_blur, k_blur), 0)

    # 3. UMBRALIZACIÓN ADAPTATIVA (OPTIMIZADA)
    # Aumentamos BlockSize a 6.0% para capturar la diferencia de objeto vs fondo.
    block_size = obtener_kernel_impar(dim_min * 0.06) 
    C = 6 # Aumentamos C para un umbral más estricto
    
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, 
        C
    )
    
    print(f"Parámetros Adaptativos Corregidos: BlockSize={block_size} (6.0%), C={C}")

    # 4. MORFOLOGÍA (CORRECCIÓN CRÍTICA DE OBJETOS)
    
    # Cierre: Usamos un kernel grande (2.5%) para asegurar que todos los huecos y bordes rotos se unan.
    k_close = obtener_kernel_impar(dim_min * 0.025) 
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # Apertura: Limpieza final de ruido (kernel pequeño)
    k_open = obtener_kernel_impar(dim_min * 0.003)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    img_clean = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 5. DETECCIÓN Y FILTRADO
    contours, _ = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_validos = []
    img_resultados = img_original.copy()
    
    # Filtro de Área (Más permisivo)
    min_area = area_total * 0.0005 # Reducido a 0.05%
    max_area = area_total * 0.20
    
    print(f"Filtro de Área Corregido: {int(min_area)} - {int(max_area)} px")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0: continue
            
            # Circularidad: Relajado a 0.3 para aceptar dados y monedas inclinadas
            circularidad = 4 * np.pi * (area / (perimetro * perimetro))
            
            if circularidad > 0.3:
                contornos_validos.append(cnt)
                
                # Visualización
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(img_resultados, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.putText(img_resultados, f"#{len(contornos_validos)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"\nRESULTADOS V6:")
    print(f"  - Objetos Detectados: {len(contornos_validos)}")
    
    # 6. VISUALIZACIÓN
    imagenes = [img_clahe, img_thresh, img_closed, img_clean, img_resultados]
    titulos = [
        '1. CLAHE', 
        f'2. Adaptive Thresh (Block={block_size})', 
        f'3. Morph Close (K={k_close}, 3 iters)', 
        '4. Limpieza Final', 
        f'5. Resultado ({len(contornos_validos)} objs)'
    ]
    
    mostrar_imagenes(imagenes, titulos, filas=2, cols=3)


segmentar_monedas_robustez('Monedas.jpg')