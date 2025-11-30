# --- Cargo imagen ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nada(val):
    """
    Esta función no hace nada. 
    Es un placeholder requerido por cv2.createTrackbar.
    """
    pass

# --- 1. Cargar la imagen ---
# Asegúrate de que 'img01_gris.jpg' esté en la misma carpeta
imagen_path = 'img01_gris.png'
img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: No se pudo cargar la imagen en '{imagen_path}'")
else:
    # --- 2. Crear ventanas y el trackbar ---
    cv2.namedWindow('Imagen Umbralada')
    cv2.namedWindow('Original')
    
    # Crear un trackbar (deslizador) para ajustar el umbral
    # (Nombre, Ventana, Valor inicial, Valor máximo, Función callback)
    cv2.createTrackbar('Umbral', 'Imagen Umbralada', 127, 255, nada)
    
    # Mostrar la imagen original como referencia
    cv2.imshow('Original', img)

    # --- 3. Bucle interactivo ---
    while True:
        
        # Obtener el valor actual del trackbar
        umbral_valor = cv2.getTrackbarPos('Umbral', 'Imagen Umbralada')
        
        # Aplicar el umbralado (threshold)
        # cv2.threshold(imagen, valor_umbral, valor_maximo, tipo_umbral)
        # THRESH_BINARY crea una imagen blanco (255) y negro (0) absoluto.
        ret, img_umbralada = cv2.threshold(img, umbral_valor, 255, cv2.THRESH_BINARY)
        
        # Mostrar la imagen resultante
        cv2.imshow('Imagen Umbralada', img_umbralada)
        
        # Esperar 1ms por una tecla. k almacena la tecla presionada.
        k = cv2.waitKey(1) & 0xFF
        
        # Condición de salida: si la tecla es 'f'
        if k == ord('f'):
            print(f"Saliendo... (Umbral final seleccionado: {umbral_valor})")
            break
            
        # Opcional: Salir si se cierra la ventana manualmente
        try:
            if cv2.getWindowProperty('Imagen Umbralada', cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # La ventana fue cerrada
            break

    # --- 4. Limpiar ---
    cv2.destroyAllWindows()