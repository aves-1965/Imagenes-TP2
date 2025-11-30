import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image):
    """Preprocesamiento: conversión a escala de grises y filtrado"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro bilateral para reducir ruido manteniendo bordes
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray, blurred

def detect_coins(image, blurred):
    """Detección de monedas usando Transformada de Hough para círculos"""
    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Transformada de Hough para círculos
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Distancia mínima entre centros
        param1=50,   # Umbral para Canny
        param2=30,   # Umbral de acumulación
        minRadius=15, # Radio mínimo
        maxRadius=50  # Radio máximo
    )
    
    coins = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            coins.append({
                'center': (x, y),
                'radius': r,
                'type': 'coin',
                'area': np.pi * r * r
            })
    
    return coins, edges

def detect_dice(image, blurred):
    """Detección de dados usando detección de contornos rectangulares"""
    # Umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Encontrar contornos
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CONTOUR_APPROX_SIMPLE
    )
    
    dice = []
    for contour in contours:
        # Calcular área y perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filtrar por tamaño (dados son más grandes que puntos)
        if area < 1000 or area > 5000:
            continue
        
        # Aproximar el contorno a un polígono
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Verificar si es aproximadamente cuadrado (4 vértices)
        if len(approx) == 4:
            # Calcular el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Verificar relación de aspecto cercana a 1 (cuadrado)
            if 0.8 <= aspect_ratio <= 1.2:
                # Calcular circularidad (4π*área/perímetro²)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Los cuadrados tienen circularidad menor que círculos
                if circularity < 0.85:
                    dice.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'type': 'dice',
                        'area': area,
                        'contour': approx
                    })
    
    return dice

def classify_by_color(image, coins):
    """Clasificar monedas por color (plateadas, doradas, bicolores)"""
    for coin in coins:
        x, y = coin['center']
        r = coin['radius']
        
        # Crear máscara circular
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Extraer región de la moneda
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        # Calcular color promedio en espacio HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask == 255]
        
        if len(pixels) > 0:
            avg_hue = np.mean(pixels[:, 0])
            avg_sat = np.mean(pixels[:, 1])
            avg_val = np.mean(pixels[:, 2])
            
            # Clasificación por color
            if avg_sat < 50:  # Baja saturación = plateada
                coin['color'] = 'plateada'
            elif 15 <= avg_hue <= 35:  # Rango amarillo-dorado
                coin['color'] = 'dorada'
            else:
                # Analizar si es bicolor (variación en anillo exterior vs centro)
                center_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.circle(center_mask, (x, y), r//2, 255, -1)
                center_pixels = hsv[center_mask == 255]
                
                if len(center_pixels) > 0:
                    center_sat = np.mean(center_pixels[:, 1])
                    if abs(avg_sat - center_sat) > 30:
                        coin['color'] = 'bicolor'
                    else:
                        coin['color'] = 'plateada'
        else:
            coin['color'] = 'desconocida'
    
    return coins

def remove_duplicates(objects, min_distance=20):
    """Eliminar detecciones duplicadas basándose en proximidad"""
    if len(objects) == 0:
        return objects
    
    filtered = []
    used = set()
    
    for i, obj in enumerate(objects):
        if i in used:
            continue
        
        center1 = obj['center']
        is_duplicate = False
        
        for j, other in enumerate(objects[i+1:], start=i+1):
            if j in used:
                continue
            
            center2 = other['center']
            distance = np.sqrt((center1[0] - center2[0])**2 + 
                             (center1[1] - center2[1])**2)
            
            if distance < min_distance:
                # Mantener el de mayor área
                if obj.get('area', 0) >= other.get('area', 0):
                    used.add(j)
                else:
                    is_duplicate = True
                    used.add(i)
                    break
        
        if not is_duplicate:
            filtered.append(obj)
    
    return filtered

def draw_results(image, coins, dice):
    """Dibujar resultados en la imagen"""
    result = image.copy()
    
    # Dibujar monedas
    for idx, coin in enumerate(coins, start=1):
        x, y = coin['center']
        r = coin['radius']
        color = coin.get('color', 'desconocida')
        
        # Color de dibujo según tipo de moneda
        if color == 'dorada':
            draw_color = (0, 215, 255)  # Dorado
        elif color == 'bicolor':
            draw_color = (0, 165, 255)  # Naranja
        else:
            draw_color = (255, 255, 255)  # Blanco (plateada)
        
        cv2.circle(result, (x, y), r, draw_color, 2)
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(result, str(idx), (x - 10, y - r - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Dibujar dados
    for idx, die in enumerate(dice, start=len(coins)+1):
        x, y, w, h = die['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, str(idx), (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result

def analyze_image(image_path):
    """Función principal de análisis"""
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    
    print("=== ANÁLISIS DE PROCESAMIENTO DE IMÁGENES DIGITALES ===\n")
    
    # 1. Preprocesamiento
    print("1. PREPROCESAMIENTO")
    gray, blurred = preprocess_image(image)
    print("   ✓ Conversión a escala de grises")
    print("   ✓ Filtrado bilateral aplicado\n")
    
    # 2. Detección de monedas
    print("2. DETECCIÓN DE MONEDAS")
    coins, edges = detect_coins(image, blurred)
    print(f"   ✓ Transformada de Hough para círculos")
    print(f"   ✓ Monedas detectadas (inicial): {len(coins)}\n")
    
    # 3. Clasificación por color
    print("3. CLASIFICACIÓN CROMÁTICA")
    coins = classify_by_color(image, coins)
    print("   ✓ Análisis en espacio HSV")
    plateadas = sum(1 for c in coins if c.get('color') == 'plateada')
    doradas = sum(1 for c in coins if c.get('color') == 'dorada')
    bicolores = sum(1 for c in coins if c.get('color') == 'bicolor')
    print(f"   - Plateadas: {plateadas}")
    print(f"   - Doradas: {doradas}")
    print(f"   - Bicolores: {bicolores}\n")
    
    # 4. Detección de dados
    print("4. DETECCIÓN DE DADOS")
    dice = detect_dice(image, blurred)
    print(f"   ✓ Detección de contornos rectangulares")
    print(f"   ✓ Dados detectados: {len(dice)}\n")
    
    # 5. Eliminación de duplicados
    print("5. VALIDACIÓN Y FILTRADO")
    coins = remove_duplicates(coins)
    dice = remove_duplicates(dice)
    print(f"   ✓ Duplicados eliminados")
    print(f"   ✓ Monedas finales: {len(coins)}")
    print(f"   ✓ Dados finales: {len(dice)}\n")
    
    # 6. Resultados finales
    print("=" * 50)
    print(f"TOTAL DE OBJETOS DETECTADOS: {len(coins) + len(dice)}")
    print(f"  • Monedas: {len(coins)}")
    print(f"  • Dados: {len(dice)}")
    print("=" * 50)
    
    # Dibujar resultados
    result = draw_results(image, coins, dice)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    # Escala de grises
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Escala de Grises')
    axes[0, 1].axis('off')
    
    # Detección de bordes
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Detección de Bordes (Canny)')
    axes[1, 0].axis('off')
    
    # Resultado final
    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Resultado: {len(coins)} monedas + {len(dice)} dados')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('resultado_analisis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Imagen de resultado guardada como 'resultado_analisis.png'")
    plt.show()
    
    return coins, dice, result

# Uso del programa
# if __name__ == "__main__":
    # Reemplazar con la ruta de tu imagen
image_path = "dados&monedas.png"
    
coins, dice, result = analyze_image(image_path)
