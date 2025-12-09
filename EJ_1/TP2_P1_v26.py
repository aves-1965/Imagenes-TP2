import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Defininimos función para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

# Funcion para rellenar
def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)  # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 

# ----------------------------------------------------------------------------------------------------------
# --- Clasificación monedas/dados --------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# --- Cargp imagen -----------------------------------------
src_path = "dados&monedas.png"
img = cv2.imread(src_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img, title='Original')

# --- Paso a escala de grises ------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(img_gray, title='Escala de grises')

"""
En este punto se probaron varios métodos de filtrado (gaussiano, etc) y ninguno dió un resultado satisfactorio.
Se decidió implementar un método de "aplanado" del fondo (Bottomless) basado en una regresión lineal simple
de los valores de intensidad a lo largo del eje horizontal (x).
Por observacion de la imagen se tomaron dos filas que representan bien el fondo (sin objetos):
- Fila 100 (x=0) y Fila 1300 (x=width)

"""

# Análisis de colores.
# Acá intentamos representar la distribución de colores en la imagen original para ver si hay 
# algún patrón que sea propio de los objetos de interés (monedas y dados) y nos permita separarlos mejor del fondo.

red_channel = img[:, :, 0].flatten()
green_channel = img[:, :, 1].flatten()
blue_channel = img[:, :, 2].flatten()

# Histograma de intensidades para cada canal de color
plt.figure(figsize=(10, 5))
plt.hist(red_channel, bins=256, range=(0, 256), density=True, alpha=0.5, color='red', label='Canal Rojo')
plt.hist(green_channel, bins=256, range=(0, 256), density=True, alpha=0.5, color='green', label='Canal Verde')
plt.hist(blue_channel, bins=256, range=(0, 256), density=True, alpha=0.5, color='blue', label='Canal Azul')

plt.title('Distribución de Intensidad de Píxeles por Canal de Color')
plt.xlabel('Intensidad de Píxel (0-255)')
plt.ylabel('Frecuencia Normalizada')
plt.grid(axis='y', alpha=0.75)
plt.xlim(0, 255)
plt.legend()
plt.show(block=False)

# La expextativa era encontrar picos en los histogramas que correspondieran a los colores 
# predominantes de las monedas y dados.
# Sin embargo, la distribución de colores es bastante uniforme, lo que indica que no hay 
# colores dominantes claros que puedan ser utilizados para segmentar los objetos de interés.
# Esto sugiere que los objetos de interés no se destacan significativamente del 
# fondo en términos de color.
# Abandonamos esta línea de análisis y seguimos con el método Bottomless. (No tiene CopyRight :-)

# ---------------- "Nivelando la cancha" (Bottomless) -----------------------
# Elegimos dos fias que sean representativas de la distribución de intensidad del fondo en 
# escala de grises. Probamos cos filas que no tengan objetos (monedas/dados) y que estén
# relativamente alejadas entre sí para captar la variación del fondo.

fila_100 = 100 # Fila específica solicitada por el usuario
fila_1300 = 1300 # Fila específica solicitada por el usuario

pixeles_100 = img[fila_100, :, 0]
pixeles_1300 = img[fila_1300, :, 1]

plt.figure(figsize=(8, 4))
plt.plot(pixeles_100, color='red', label='Fila 100')
plt.plot(pixeles_1300, color='green', label='Fila 1300')

plt.title(f'Valores de los píxeles de la filas 100 y 1300')
plt.xlabel('Posición en el eje X (píxeles)')
plt.ylabel('Intensidad del píxel (0-255)')
plt.xlim(0, img.shape[1] - 1) # Ajustar el límite X al ancho de la imagen
plt.grid(True)
plt.legend()
plt.show(block=False)

x_100 = np.arange(len(pixeles_100)).reshape(-1, 1)
y_100 = pixeles_100.reshape(-1, 1)

model_100 = LinearRegression()
model_100.fit(x_100, y_100)

slope_100 = model_100.coef_[0][0]
intercept_100 = model_100.intercept_[0]

# --------------------------------------------------------------------------------
print(f'Ecuación para pixeles_100: y = {slope_100:.4f} * x + {intercept_100:.4f}')
# --------------------------------------------------------------------------------

x_1300 = np.arange(len(pixeles_1300)).reshape(-1, 1)
y_1300 = pixeles_1300.reshape(-1, 1)

model_1300 = LinearRegression()
model_1300.fit(x_1300, y_1300)

slope_1300 = model_1300.coef_[0][0]
intercept_1300 = model_1300.intercept_[0]

# ------------------------------------------------------------------------
print(f'Ecuación para pixeles_1300: y = {slope_1300:.4f} * x + {intercept_1300:.4f}')
# ------------------------------------------------------------------------

# Ahora buscamos una recta resultante que represente mejor el fondo
# Combinamos los datos de ambas filas
x_combined = np.vstack((x_100, x_1300))
y_combined = np.vstack((y_100, y_1300))

# Nueva regresión lineal con los datos combinados
model_fondo = LinearRegression()
model_fondo.fit(x_combined, y_combined)

# Pendiente y ordenada al origen de la recta combinada
slope_fondo = model_fondo.coef_[0][0]
intercept_fondo = model_fondo.intercept_[0]

# ------------------------------------------------------------------------
print(f'Ecuación para pixeles_fondo (combinado): y = {slope_fondo:.4f} * x + {intercept_fondo:.4f}')
# ------------------------------------------------------------------------

# Veamos que resultó de todo ésto.
plt.figure(figsize=(8, 4))
# Pixeles Originales fila 100
plt.plot(np.arange(len(pixeles_100)), pixeles_100, color='blue', label='Fila 100 Gris (Original)', alpha=0.7)
# Pixeles Recta fila 100
plt.plot(x_100, model_100.predict(x_100), color='cyan', linestyle='--', label=f'Regresión Fila 100 (y = {slope_100:.2f}x + {intercept_100:.2f})')
# Pixeles Originales fila 1300
plt.plot(np.arange(len(pixeles_1300)), pixeles_1300, color='red', label='Fila 1300 Gris (Original)', alpha=0.7)
# Pixeles Recta fila 1300
plt.plot(x_1300, model_1300.predict(x_1300), color='magenta', linestyle='--', label=f'Regresión Fila 1300 (y = {slope_1300:.2f}x + {intercept_1300:.2f})')
# Recta Fondo Combinado
x_full_width = np.arange(img_gray.shape[1]).reshape(-1, 1)
y_pred_fondo = model_fondo.predict(x_full_width)

plt.plot(x_full_width, y_pred_fondo, color='green', linewidth=2, label=f'Regresión Fondo (y = {slope_fondo:.2f}x + {intercept_fondo:.2f})')
plt.title('Valores de los píxeles de las filas 100 y 1300 en Gris con Rectas de Regresión')
plt.xlabel('Posición en el eje X (píxeles)')
plt.ylabel('Intensidad del píxel (0-255)')
plt.xlim(0, img_gray.shape[1] - 1)
plt.grid(True)
plt.legend()
plt.show(block=False)

# Ahora restamos la superficie estimada del fondo a toda la imagen en gris
# para "aplanar" el fondo y facilitar la detección de bordes.
h, w = img_gray.shape
x_coords = np.arange(w)

a = 0.06 # Pendiente
b = 40.47 # Ordenada al Origen

y_pred_surface_user = slope_fondo * x_coords + b
# y_pred_surface_user = a * x_coords + b

img_bottomless = img_gray.astype(np.float64) - y_pred_surface_user
imshow(img_bottomless, title='Imagen en Gris con "aplanado" del Fondo.')

# Con esta imagen comenzamos con el procesamiento habitual: bordes, morfología, etc.
# --- Detección de bordes con Canny ------------------------
# Preparación:
# 1. NORM_MINMAX: Toma el valor más bajo de la matriz y lo mapea a 'alpha' (0).
# 2. Toma el valor más alto y lo mapea a 'beta' (255).
# 3. dtype=cv2.CV_8U: Convierte el resultado final al tipo entero de 8 bits.

img4canny = cv2.normalize(src=img_bottomless, dst=None, alpha=0, beta=255, 
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

th1= 77     # Valor óptimo encontrado por prueba y error: 87
th2= 175    # Valor óptimo encontrado por prueba y error: 175
# Estos valores se determinaron con sucesivas pruebas visuales buscando el mejor compromiso
# entre detección de bordes y ruido.

edges = cv2.Canny(img4canny, th1, th2)
titulo = f'Canny con previo filtrado Bottomless Normalizado (th1={th1}, th2={th2})'
imshow(edges, title=titulo)


# --- Dilato -------------------------------------------
img_dilatada = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
imshow(img_dilatada, title='Dilatación de Bordes con Elemento Elíptico')

# --- Relleno de los objetos -------------------------------
img_rellena = fillhole(img_dilatada)
imshow(img_rellena, title='Rellenado de Objetos post-Dilatación')

# --- Erosion ----------------------------------------------
img_erode = cv2.erode(img_rellena, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
imshow(img_erode, title='Erosion para Refinar Objetos')

# Apertura adicional (si es necesario) -----------------------
# Eliminamos pequeños ruidos que hayan quedado
img_abierta = cv2.morphologyEx(img_erode, cv2.MORPH_OPEN, 
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
imshow(img_abierta, title='Apertura Adicional para Limpiar Ruido')

# --- Detección de objetos -----------------------------------------------
# El rho (forma) será el criterio principal para discriminar entre monedas y dados.
# Se determinó empíricamente que un valor de rho > 0.80 corresponde a monedas.
num_labels, labels = cv2.connectedComponents(img_abierta)
mask_objetos = np.zeros_like(img)
RHO_TH = 0.80

# --- Cálculo del umbral de área mínima (0.48% del total) ---
# Este umbra se estimó como el mínimo porcentaje de a superficie total de la imagen que debe 
# tener un objeto para ser considerado válido.
area_total_imagen = img.shape[0] * img.shape[1]
area_minima = area_total_imagen * 0.004

# --- NUEVO: Inicializamos contadores para el reporte ---
cant_dados = 0
cant_monedas_chicas = 0
cant_monedas_medianas = 0
cant_monedas_grandes = 0

for i in range(1, num_labels):
        # -- Analizo cada objeto --
        obj = (labels == i).astype(np.uint8) * 255
        contour, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contour) == 0: continue
            
        cnt = contour[0]
        
        # --- (Opcional) Aproximación de Polígonos ---
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        # ----------------------------------------------------------

        area = cv2.contourArea(cnt)
        # --- NUEVO: Filtro por superficie relativa ---
        # Si el área es menor al 0.48% de la imagen, descartamos el objeto
        if area < area_minima:
            continue
        # ---------------------------------------------
        perimetro = cv2.arcLength(cnt, True)
        
        if perimetro == 0: 
            rho = 0
        else:
            rho = 4 * np.pi * (area / (perimetro ** 2))

        tipo_objeto = "Desconocido"
        
        # --- Lógica de clasificación MODIFICADA ---
        if rho > RHO_TH:
            # Es una Moneda. Ahora discriminamos por perímetro.
            if perimetro < 485:
                # Moneda Chica -> Pintamos VERDE (Canal 1 en RGB)
                mask_objetos[:,:,1] = np.maximum(mask_objetos[:,:,1], obj)
                tipo_objeto = "Moneda Chica"
                cant_monedas_chicas += 1
            elif perimetro < 560:
                # Moneda Mediana -> Pintamos AMARILLO (Canal 0 en RGB)
                mask_objetos[:,:,0] = np.maximum(mask_objetos[:,:,0], obj) # Agrego Rojo
                mask_objetos[:,:,1] = np.maximum(mask_objetos[:,:,1], obj) # Agrego Verde
                tipo_objeto = "Moneda Mediana"
                cant_monedas_medianas += 1
            else:
                # Moneda Grande -> Pintamos ROJO (Canal 0 en RGB)
                mask_objetos[:,:,0] = np.maximum(mask_objetos[:,:,0], obj)
                tipo_objeto = "Moneda Grande"
                cant_monedas_grandes += 1


        else:
            # Es Cuadrado/Dado -> Pintamos AZUL (Canal 2 en RGB)
            mask_objetos[:,:,2] = np.maximum(mask_objetos[:,:,2], obj)
            tipo_objeto = "Dado"
            cant_dados += 1

        # Visualización del ID en la imagen
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(mask_objetos, str(i), (cX - 10, cY + 5), 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# --- Imprimir reporte en consola ---
print("\n" + "="*30)
print("       REPORTE FINAL       ")
print("="*30)
print(f"Dados encontrados       : {cant_dados}")
print(f"Monedas chicas (<485px) : {cant_monedas_chicas}")
print(f"Monedas medianas (485-560px): {cant_monedas_medianas}")
print(f"Monedas grandes (>485px): {cant_monedas_grandes}")
print(f"TOTAL OBJETOS           : {cant_dados + cant_monedas_chicas + cant_monedas_medianas + cant_monedas_grandes}")
print("="*30 + "\n")

imshow(mask_objetos, title='Mascara Clasificada (Rojo=Grande, Verde=Chica, Azul=Dado)', color_img=True)

# --- Muestro resultado sobre imagen original --------------------
alpha = 0.5
img_resaltada = cv2.addWeighted(mask_objetos, alpha, img, 1 - alpha, 0)
imshow(img_resaltada, title='Resultado Final', color_img=True)

# ----------------------------------------------------------------------------------------------------------
# --- Detección de puntos en dados ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# ============================================
# PARÁMETROS AJUSTABLES
# ============================================

# 1. DETECCIÓN DE MARCO
UMBRAL_MARCO = 30  # Rango: 20-40. Menor = detecta más oscuro

# 2. DETECCIÓN DE DADOS
UMBRAL_DADOS = 50  # Rango: 30-80. Ajusta según brillo del dado
KERNEL_DADOS_SIZE = 7  # Rango: 5-9. Mayor = más limpieza, puede unir puntos cercanos

# 3. FILTRADO DE DADOS
AREA_MIN_DADO = 2000  # Rango: 1000-3000. Mínimo tamaño de dado en píxeles²
RATIO_MIN_DADO = 0.6   # Rango: 0.5-0.7. Proporción mínima ancho/alto
RATIO_MAX_DADO = 1.6   # Rango: 1.4-1.8. Proporción máxima ancho/alto
LADO_MIN_DADO = 50     # Rango: 40-70. Lado mínimo del dado en píxeles

# 4. DETECCIÓN DE PUNTOS (CRÍTICO PARA EVITAR FALSOS POSITIVOS)
FACTOR_UMBRAL_PUNTOS = 0.75  # Rango: 0.5-0.8. Mayor = más estricto, detecta menos
UMBRAL_MAX_PUNTOS = 70       # Rango: 50-70. Máximo umbral para puntos

# 5. MORFOLOGÍA DE PUNTOS
KERNEL_PUNTOS_SIZE = 3  # Rango: 1-3. Mayor = más limpieza pero puede eliminar puntos pequeños

# 6. FILTRADO DE PUNTOS (MUY IMPORTANTE)
AREA_MIN_PUNTO = 30      # Rango: 10-25. Mínimo tamaño de punto en píxeles²
AREA_MAX_PUNTO = 200     # Rango: 200-500. Máximo tamaño de punto en píxeles²
                         # REDUCIR ESTE VALOR EVITA DETECTAR ÁREAS GRANDES COMO PUNTOS

CIRCULARIDAD_MIN = 0.5  # Rango: 0.2-0.5. Mayor = más estricto con la forma circular
                         # 1.0 = círculo perfecto, 0.0 = línea
                         # AUMENTAR ESTE VALOR EVITA DETECTAR FORMAS IRREGULARES

# ============================================
# CÓDIGO PRINCIPAL - DETECCIÓN DE PUNTOS
# ============================================

# PASO 1: Crear máscara que contenga SOLO los dados (canal azul de mask_objetos)
# Los dados fueron pintados en azul (canal 2) durante la clasificación
mascara_solo_dados = mask_objetos[:, :, 2]
imshow(mascara_solo_dados, title='Paso 1: Máscara con SOLO dados detectados')

# PASO 2: Crear imagen gris LIMPIA que contenga SOLO las regiones de los dados
# 
img_solo_dados = cv2.bitwise_and(img_gray, img_gray, mask=mascara_solo_dados)
imshow(img_solo_dados, title='Paso 2: Imagen Gris LIMPIA solo con dados (fondo negro)')

# Encontrar todos los contornos en la máscara de dados
contornos_temp, _ = cv2.findContours(mascara_solo_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear lista de contornos con su área
contornos_con_area = [(contorno, cv2.contourArea(contorno)) for contorno in contornos_temp]

# Ordenar por área de mayor a menor
contornos_con_area.sort(key=lambda x: x[1], reverse=True)

# Quedarnos solo con los 2 más grandes
contornos_grandes = [c[0] for c in contornos_con_area[:2]]

# Crear nueva máscara con solo los 2 dados más grandes
mascara_solo_dados = np.zeros_like(mascara_solo_dados)
cv2.drawContours(mascara_solo_dados, contornos_grandes, -1, 255, -1)

# Actualizar img_solo_dados con la nueva máscara
img_solo_dados = cv2.bitwise_and(img_gray, img_gray, mask=mascara_solo_dados)
imshow(img_solo_dados, title='Paso 2B: Solo los 2 dados más grandes')


# PASO 3: Encontrar contornos de cada dado individual en la máscara
contornos_dados, _ = cv2.findContours(mascara_solo_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Dados detectados para análisis de puntos: {len(contornos_dados)}\n")

# Crear imagen resultado para visualización
img_resultado_dados = img.copy()

# PASO 3: Aplicar blur a la imagen ORIGINAL en gris (NO a img_solo_dados)
gris_blur = cv2.GaussianBlur(img_solo_dados, (5, 5), 0)
imshow(gris_blur, title='Paso 3: Imagen original con Gaussian Blur aplicado')


# PASO 4: Procesar cada dado individualmente (estrategia de puntos_dados_2.py)
for idx, contorno_dado in enumerate(contornos_dados):
    # Obtener bounding box del dado
    x, y, w, h = cv2.boundingRect(contorno_dado)
    
    # Extraer ROI del dado desde la imagen original con blur
    roi = gris_blur[y:y+h, x:x+w]
    
    # Detectar puntos - igual que en puntos_dados_2.py
    media_roi = roi.mean()
    umbral_puntos_valor = min(UMBRAL_MAX_PUNTOS, media_roi * FACTOR_UMBRAL_PUNTOS)
    _, mascara_puntos = cv2.threshold(roi, umbral_puntos_valor, 255, cv2.THRESH_BINARY_INV)
    
    # Operaciones morfológicas para limpiar
    kernel_small = np.ones((KERNEL_PUNTOS_SIZE, KERNEL_PUNTOS_SIZE), np.uint8)
    mascara_puntos = cv2.morphologyEx(mascara_puntos, cv2.MORPH_OPEN, kernel_small)
    mascara_puntos = cv2.morphologyEx(mascara_puntos, cv2.MORPH_CLOSE, kernel_small)
    imshow(mascara_puntos, title='Operaciones morfológicas para limpiar')
    
    # Encontrar contornos de puntos candidatos
    contornos_puntos, _ = cv2.findContours(mascara_puntos, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    
    print(f"  Dado {idx+1}: Encontrados {len(contornos_puntos)} contornos candidatos")
    
    # Filtrar puntos válidos por área y circularidad
    puntos_validos = []
    for contorno_punto in contornos_puntos:
        area_punto = cv2.contourArea(contorno_punto)
        
        perimetro = cv2.arcLength(contorno_punto, True)
        circularidad = 4 * np.pi * area_punto / (perimetro * perimetro) if perimetro > 0 else 0
        # Calcular circularidad escalada (multiplicada por 1000 para comparación)
        circularidad_escalada = circularidad * 1000
        print(f"    Contorno: área={area_punto:.1f}, perímetro={perimetro:.1f}, circularidad={circularidad_escalada:.0f}")
        
        # ATENCION CLAUDE! EL ATRIBUTO MAS IMPORTANTE PARA ENCONTRAR PUNTOS ES LA CIRCULAIRIDAD. SI ES MAYOR A 600 ES UN PUNTO. NO EMPLEAR AREA PARA DETERMINAR UN PUNTO.
        
        if circularidad_escalada > 600:
            M = cv2.moments(contorno_punto)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                puntos_validos.append((cx, cy))
    
    # Dibujar resultados en la imagen original
    cv2.rectangle(img_resultado_dados, (x, y), (x+w, y+h), (255, 0, 0), 3)
    
    for px, py in puntos_validos:
        cv2.circle(img_resultado_dados, (px, py), 12, (0, 255, 0), 2)
        cv2.circle(img_resultado_dados, (px, py), 3, (0, 0, 255), -1)
    
    num_puntos = len(puntos_validos)
    texto = f'Dado {idx+1}: {num_puntos} pts'
    cv2.putText(img_resultado_dados, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    print(f"Dado {idx+1}: {num_puntos} puntos detectados")

# Mostrar resultado final con puntos
imshow(img_resultado_dados, title='Detección de Puntos en Dados', color_img=True)