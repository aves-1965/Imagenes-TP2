import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagenes(imagenes, titulos, filas=2, cols=3, figsize=(15, 10)):
    """Función auxiliar para mostrar múltiples imágenes"""
    fig, axes = plt.subplots(filas, cols, figsize=figsize)
    axes = axes.ravel()
    
    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        if len(img.shape) == 2:  # Imagen en escala de grises
            axes[i].imshow(img, cmap='gray')
        else:  # Imagen en color
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(titulo)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def segmentar_monedas_dados(ruta_imagen):
    """
    Segmenta automáticamente monedas y dados de una imagen con fondo no uniforme
    
    Args:
        ruta_imagen: Ruta al archivo de imagen
    
    Returns:
        Diccionario con resultados del procesamiento
    """
    print("="*60)
    print("PUNTO A: SEGMENTACIÓN AUTOMÁTICA DE MONEDAS Y DADOS")
    print("="*60)
    
    # ========== ETAPA 1: CARGA DE IMAGEN ==========
    print("\n[ETAPA 1] Cargando imagen...")
    img_original = cv2.imread(ruta_imagen)
    
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'")
        return None
    
    print(f"  ✓ Imagen cargada exitosamente")
    print(f"  ✓ Dimensiones: {img_original.shape[1]}x{img_original.shape[0]} píxeles")
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # ========== ETAPA 2: CORRECCIÓN DE ILUMINACIÓN NO UNIFORME ==========
    print("\n[ETAPA 2] Corrigiendo iluminación no uniforme...")
    
    # Aplicar desenfoque gaussiano para estimar el fondo
    blur_fondo = cv2.GaussianBlur(img_gray, (51, 51), 0)
    
    # Restar el fondo estimado para compensar iluminación
    img_corregida = cv2.subtract(blur_fondo, img_gray)
    
    print("  ✓ Fondo estimado mediante filtro gaussiano (kernel 51x51)")
    print("  ✓ Iluminación corregida por sustracción")
    
    # ========== ETAPA 3: UMBRALIZACIÓN ADAPTATIVA ==========
    print("\n[ETAPA 3] Aplicando umbralización...")
    
    # Umbralización de Otsu
    _, img_thresh = cv2.threshold(img_corregida, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"  ✓ Método: Otsu (umbral automático)")
    
    # ========== ETAPA 4: OPERACIONES MORFOLÓGICAS ==========
    print("\n[ETAPA 4] Refinando segmentación con morfología...")
    
    # Crear elementos estructurantes
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    # Opening para eliminar ruido pequeño
    img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_opening)
    
    # Closing para cerrar huecos dentro de objetos
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing)
    
    print("  ✓ Opening aplicado (kernel elíptico 5x5) - elimina ruido")
    print("  ✓ Closing aplicado (kernel elíptico 11x11) - cierra huecos")
    
    # ========== ETAPA 5: DETECCIÓN DE CONTORNOS ==========
    print("\n[ETAPA 5] Detectando contornos de objetos...")
    
    contours, hierarchy = cv2.findContours(img_closing, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  ✓ Contornos detectados: {len(contours)}")
    
    # ========== ETAPA 6: FILTRADO DE CONTORNOS ==========
    print("\n[ETAPA 6] Filtrando contornos por área...")
    
    # Calcular área mínima basada en el tamaño de la imagen
    area_imagen = img_original.shape[0] * img_original.shape[1]
    area_minima = area_imagen * 0.0005  # 0.05% del área total
    area_maxima = area_imagen * 0.1     # 10% del área total
    
    contornos_filtrados = []
    areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_minima < area < area_maxima:
            contornos_filtrados.append(contour)
            areas.append(area)
    
    print(f"  ✓ Área mínima considerada: {area_minima:.0f} px²")
    print(f"  ✓ Área máxima considerada: {area_maxima:.0f} px²")
    print(f"  ✓ Objetos válidos detectados: {len(contornos_filtrados)}")
    
    # ========== ETAPA 7: VISUALIZACIÓN DE RESULTADOS ==========
    print("\n[ETAPA 7] Generando visualización de resultados...")
    
    # Crear imagen con contornos dibujados
    img_contornos = img_original.copy()
    cv2.drawContours(img_contornos, contornos_filtrados, -1, (0, 255, 0), 3)
    
    # Crear máscaras individuales y imagen segmentada
    img_segmentada = np.zeros_like(img_original)
    mascara_total = np.zeros(img_gray.shape, dtype=np.uint8)
    
    for i, contour in enumerate(contornos_filtrados):
        # Crear máscara para este objeto
        mascara = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [contour], -1, 255, -1)
        mascara_total = cv2.bitwise_or(mascara_total, mascara)
        
        # Extraer objeto de la imagen original
        img_segmentada = cv2.bitwise_or(img_segmentada, 
                                        cv2.bitwise_and(img_original, img_original, 
                                                       mask=mascara))
        
        # Dibujar número de objeto y bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_contornos, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_contornos, f"#{i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    print(f"  ✓ Objetos etiquetados del #1 al #{len(contornos_filtrados)}")
    
    # ========== VISUALIZACIÓN ==========
    print("\n[VISUALIZACIÓN] Mostrando resultados del procesamiento...")
    
    imagenes = [
        img_original,
        img_gray,
        img_corregida,
        img_thresh,
        img_closing,
        img_contornos
    ]
    
    titulos = [
        'Imagen Original',
        'Escala de Grises',
        'Corrección de Iluminación',
        'Umbralización (Otsu)',
        'Morfología (Opening + Closing)',
        f'Segmentación Final ({len(contornos_filtrados)} objetos)'
    ]
    
    mostrar_imagenes(imagenes, titulos, filas=2, cols=3, figsize=(18, 12))
    
    # Mostrar imagen segmentada por separado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(mascara_total, cmap='gray')
    axes[0].set_title('Máscara de Segmentación')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Objetos Segmentados')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*60)
    print("RESUMEN DE SEGMENTACIÓN")
    print("="*60)
    print(f"Total de objetos segmentados: {len(contornos_filtrados)}")
    print(f"Área promedio: {np.mean(areas):.0f} px²")
    print(f"Área mínima detectada: {np.min(areas):.0f} px²")
    print(f"Área máxima detectada: {np.max(areas):.0f} px²")
    print("="*60)
    
    # Retornar resultados
    return {
        'imagen_original': img_original,
        'imagen_gray': img_gray,
        'imagen_corregida': img_corregida,
        'imagen_threshold': img_thresh,
        'imagen_morfologia': img_closing,
        'mascara_segmentacion': mascara_total,
        'imagen_segmentada': img_segmentada,
        'imagen_contornos': img_contornos,
        'contornos': contornos_filtrados,
        'areas': areas,
        'num_objetos': len(contornos_filtrados)
    }


# ========== EJECUCIÓN PRINCIPAL ==========
if __name__ == "__main__":
    # Ruta a la imagen (ajustar según corresponda)
    ruta_imagen = 'monedas.jpg'
    
    # Ejecutar segmentación
    resultados = segmentar_monedas_dados(ruta_imagen)
    
    if resultados:
        print("\n✓ Procesamiento completado exitosamente")
        print(f"✓ Se detectaron {resultados['num_objetos']} objetos")
    else:
        print("\n✗ Error en el procesamiento")