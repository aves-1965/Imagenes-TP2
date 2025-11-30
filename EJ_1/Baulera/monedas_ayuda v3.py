import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

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
# --- Detecto bordes ----------------------------------------
img_blur_median = cv2.medianBlur(img_gray, 7) # ksize = 7 genera bordes mas separados
# ksize = 5 genera mas ruidos pero bordes mas definidos
# ksize = 3 es una porqueria
# imshow(img_blur_median, title='Median Blur')

kernelito = 3
img_blur_gauss = cv2.GaussianBlur(img_gray, (kernelito, kernelito), 0)

img_bordes_median = cv2.Canny(img_blur_median, 30, 100)
imshow(img_bordes_median, title='Canny con previo filtrado por Mediana')

img_bordes_gauss = cv2.Canny(img_blur_gauss, 30, 100)
imshow(img_bordes_gauss, title='Canny con previo filtrado por Gaussiana')

"""

h, w = img_gray.shape
x_coords = np.arange(w)

a = 0.06 # Pendiente
b = 40.47 # Ordenada al Origen

y_pred_surface_user = a * x_coords + b

img_bottomless = img_gray.astype(np.float64) - y_pred_surface_user

imshow(img_bottomless, title='Imagen en Gris con "aplanado" del Fondo.')

# --- Probamos Canny con Bottomless -----------------------------------

# --- SOLUCIÓN: Normalización con cv2.normalize ---
# Esta función hace tres cosas clave aquí:
# 1. NORM_MINMAX: Toma el valor más bajo de tu matriz y lo mapea a 'alpha' (0).
# 2. Toma el valor más alto y lo mapea a 'beta' (255).
# 3. dtype=cv2.CV_8U: Convierte el resultado final al tipo de entero de 8 bits requerido.

img4canny = cv2.normalize(src=img_bottomless, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

th1= 87     # Valor óptimo encontrado por prueba y error: 87
th2= 175    # Valor óptimo encontrado por prueba y error: 
edges = cv2.Canny(img4canny, th1, th2)
titulo = f'Canny con previo filtrado Bottomless Normalizado (th1={th1}, th2={th2})'
imshow(edges, title=titulo)


# --- Dilato -------------------------------------------
img_dilatada = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
imshow(img_dilatada, title='Dilato')

# --- Relleno de los objetos -------------------------------
img_rellena = fillhole(img_dilatada)
imshow(img_rellena, title='Relleno Huecos')


# ------------ HASTA AQUÍ LLEGAMOS BIEN --------------------------------------------------
# --- Erosion ----------------------------------------------
img_erode = cv2.erode(img_rellena, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
imshow(img_erode, title='Erosion')

# Apertura adicional (si es necesario)
# img_erode = cv2.morphologyEx(img_erode, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
# imshow(img_erode, title='Apertura Adicional')

# --- Detección de objetos -----------------------------------------------
num_labels, labels = cv2.connectedComponents(img_erode)
mask_objetos = np.zeros_like(img)
RHO_TH = 0.80

# --- NUEVO: Cálculo del umbral de área mínima (0.48% del total) ---
area_total_imagen = img.shape[0] * img.shape[1]
area_minima = area_total_imagen * 0.004

# --- NUEVO: Inicializamos contadores para el reporte ---
cant_dados = 0
cant_monedas_chicas = 0
cant_monedas_grandes = 0

# --- 1. Abrimos el archivo CSV para escritura ---
with open('datos_objetos.csv', mode='w', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv, delimiter=',')
    
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
            cv2.putText(mask_objetos, str(i), (cX - 10, cY + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

print("Archivo 'datos_objetos.csv' guardado exitosamente.")

# --- NUEVO: Imprimir reporte en consola ---
print("\n" + "="*30)
print("       REPORTE FINAL       ")
print("="*30)
print(f"Dados encontrados       : {cant_dados}")
print(f"Monedas chicas (<485px) : {cant_monedas_chicas}")
print(f"Monedas grandes (>485px): {cant_monedas_grandes}")
print(f"TOTAL OBJETOS           : {cant_dados + cant_monedas_chicas + cant_monedas_grandes}")
print("="*30 + "\n")

imshow(mask_objetos, title='Mascara Clasificada (Rojo=Grande, Verde=Chica, Azul=Dado)')

# --- Muestro resultado sobre imagen original --------------------
alpha = 0.5
img_resaltada = cv2.addWeighted(mask_objetos, alpha, img, 1 - alpha, 0)
imshow(img_resaltada, title='Resultado Final')