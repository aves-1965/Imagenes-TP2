import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
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
plt.xlim(0, img_rgb.shape[1] - 1) # Ajustar el límite X al ancho de la imagen
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

# --- 1. Abrimos el archivo CSV para escritura ---
with open('datos_objetos.csv', mode='w', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv, delimiter=';')
    
    # Escribimos el encabezado (Header)
    writer.writerow(['ID_Objeto', 'Area', 'Perimetro', 'Rho', 'Clasificacion'])

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
                # Moneda Mediana -> Pintamos ROJO (Canal 0 en RGB)
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

        # -- 2. Escribir datos en el CSV --
        writer.writerow([i, round(area, 2), round(perimetro, 2), round(rho, 4), tipo_objeto])
        
        # -- 3. Visualización del ID en la imagen --
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.putText(mask_objetos, str(i), (cX - 10, cY + 5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 6)
            cv2.putText(mask_objetos, str(i), (cX - 10, cY + 5), 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

print("Archivo 'datos_objetos.csv' guardado exitosamente.")

# --- NUEVO: Imprimir reporte en consola ---
print("\n" + "="*30)
print("       REPORTE FINAL       ")
print("="*30)
print(f"Dados encontrados       : {cant_dados}")
print(f"Monedas chicas (<485px) : {cant_monedas_chicas}")
print(f"Monedas medianas (485-560px): {cant_monedas_medianas}")
print(f"Monedas grandes (>485px): {cant_monedas_grandes}")
print(f"TOTAL OBJETOS           : {cant_dados + cant_monedas_chicas + cant_monedas_medianas + cant_monedas_grandes}")
print("="*30 + "\n")

imshow(mask_objetos, title='Mascara Clasificada (Rojo=Grande, Verde=Chica, Azul=Dado)')

# --- Muestro resultado sobre imagen original --------------------
alpha = 0.5
img_resaltada = cv2.addWeighted(mask_objetos, alpha, img, 1 - alpha, 0)
imshow(img_resaltada, title='Resultado Final')