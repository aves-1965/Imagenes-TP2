import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagenes(imagenes, titulos, filas=2, cols=3, figsize=(16, 10)):
    """Función auxiliar para visualización"""
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

def segmentar_monedas_adaptativo(ruta_imagen):
    print("="*60)
    print("PROCESAMIENTO V5: UMBRALIZACIÓN ADAPTATIVA + CLAHE")
    print("="*60)
    
    # 1. CARGA
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print("Error: No se pudo cargar la imagen.")
        return

    h, w = img_original.shape[:2]
    dim_min = min(h, w)
    area_total = h * w
    print(f"Dimensiones: {w}x{h} píxeles")

    # 2. PREPROCESAMIENTO (GRAY + CLAHE)
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: Vital para resaltar la textura de las monedas frente al fondo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    # 3. SUAVIZADO (BLUR)
    # Necesario para reducir el ruido antes del threshold adaptativo
    # Tamaño: ~0.5% de la imagen (suficiente para borrar ruido de sensor)
    k_blur = obtener_kernel_impar(dim_min * 0.005)
    img_blur = cv2.GaussianBlur(img_clahe, (k_blur, k_blur), 0)

    # 4. UMBRALIZACIÓN ADAPTATIVA (CAMBIO PRINCIPAL)
    # Calcula el umbral para cada pixel basado en un entorno local (block_size).
    # Block Size: ~2% de la imagen. Lo suficientemente grande para ver "manchas"
    # pero pequeño para ignorar gradientes de luz globales.
    block_size = obtener_kernel_impar(dim_min * 0.025) # ~2.5%
    C = 4 # Constante para restar (ajuste fino de sensibilidad)
    
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # Invertido: Objetos blancos, fondo negro
        block_size, 
        C
    )
    
    print(f"Parámetros Adaptativos: BlockSize={block_size}, C={C}")

    # 5. MORFOLOGÍA MATEMÁTICA (RECONSTRUCCIÓN)
    # El threshold adaptativo deja huecos y ruido. Usamos morfología para limpiar.
    
    # Cierre (Close): Conecta puntos cercanos (rellena las monedas y dados)
    k_close = obtener_kernel_impar(dim_min * 0.015) # ~1.5% Kernel fuerte
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Apertura (Open): Elimina el ruido blanco del fondo (polvo/manchas)
    k_open = obtener_kernel_impar(dim_min * 0.005) # ~0.5%
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    img_clean = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 6. DETECCIÓN DE CONTORNOS
    contours, _ = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_validos = []
    img_resultados = img_original.copy()
    
    # Filtros de Área (Dinámicos)
    # Mínimo: 0.1% del área (evita ruido)
    # Máximo: 20% del área (evita detectar el fondo entero)
    min_area = area_total * 0.001 
    max_area = area_total * 0.20
    
    print(f"Filtro de Área: {int(min_area)} - {int(max_area)} px")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            # Filtro de Compacidad/Circularidad
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0: continue
            
            # Circularidad = 1 es un círculo perfecto. Cuadrados ~0.78.
            # Usamos > 0.3 para ser permisivos con monedas inclinadas o dados
            circularidad = 4 * np.pi * (area / (perimetro * perimetro))
            
            if circularidad > 0.3:
                contornos_validos.append(cnt)
                
                # Dibujar
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.rectangle(img_resultados, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.putText(img_resultados, f"#{len(contornos_validos)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"\nRESULTADOS V5:")
    print(f"  - Objetos Detectados: {len(contornos_validos)}")
    
    if len(contornos_validos) == 0:
        print("  ⚠ ADVERTENCIA: No se detectaron objetos. Revisa si la imagen es muy oscura.")

    # 7. VISUALIZACIÓN
    imagenes = [
        img_clahe, 
        img_blur, 
        img_thresh, 
        img_closed, 
        img_clean, 
        img_resultados
    ]
    titulos = [
        '1. CLAHE (Contraste Local)', 
        '2. Blur (Suavizado)', 
        f'3. Adaptive Thresh (Block={block_size})', 
        f'4. Morph Close (Relleno K={k_close})', 
        '5. Morph Open (Limpieza Final)', 
        f'6. Resultado ({len(contornos_validos)} objs)'
    ]
    
    mostrar_imagenes(imagenes, titulos)

segmentar_monedas_adaptativo('Monedas.jpg')