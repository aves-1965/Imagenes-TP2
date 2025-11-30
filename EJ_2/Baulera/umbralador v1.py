import cv2
import numpy as np
import csv
import os

def nada(val):
    """
    Función placeholder requerida por cv2.createTrackbar.
    """
    pass

# --- 1. Inicialización ---
resultados = [] # Lista para guardar la "tabla" de resultados
ventana_principal = 'Umbralado Interactivo (Original | Umbralado)'
cv2.namedWindow(ventana_principal, cv2.WINDOW_AUTOSIZE)

# Crear el deslizador (trackbar) en la ventana principal
cv2.createTrackbar('Umbral', ventana_principal, 127, 255, nada)

print("Iniciando el proceso de umbralado...")
print("Instrucciones:")
print("  - Ajuste el deslizador 'Umbral'.")
print("  - Presione 's' para guardar el valor y pasar a la siguiente imagen.")
print("  - Presione 'q' para salir del programa y guardar el CSV.")
print("-" * 30)

# --- 2. Bucle principal (Iterar sobre archivos) ---
for i in range(1, 13):
    # Formatear el nombre del archivo (img01.png, img02.png, ..., img12.png)
    filename = f"img{i:02d}.png"
    
    # Verificar si el archivo existe antes de intentar cargarlo
    if not os.path.exists(filename):
        print(f"\nADVERTENCIA: No se encontró '{filename}'. Saltando...")
        continue

    # Cargar la imagen en escala de grises
    img_gris = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img_gris is None:
        print(f"\nERROR: No se pudo cargar '{filename}'. Saltando...")
        continue
        
    print(f"\nProcesando: {filename}")
    
    # Convertir la original (gris) a BGR (3 canales) para poder apilarla
    # con la imagen umbralada (que también convertiremos a BGR).
    img_display_original = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2BGR)

    # --- 3. Bucle interactivo (Ajuste de umbral por imagen) ---
    while True:
        # Obtener el valor actual del deslizador
        umbral_valor = cv2.getTrackbarPos('Umbral', ventana_principal)
        
        # Aplicar el umbralado binario (blanco y negro absolutos)
        ret, img_umbralada_gris = cv2.threshold(img_gris, umbral_valor, 255, cv2.THRESH_BINARY)
        
        # Convertir la imagen umbralada (1 canal) a BGR (3 canales)
        img_display_umbralada = cv2.cvtColor(img_umbralada_gris, cv2.COLOR_GRAY2BGR)

        # Apilar las dos imágenes (Original y Umbralada) horizontalmente
        vista_comparativa = np.hstack((img_display_original, img_display_umbralada))
        
        # Mostrar la vista comparativa
        cv2.imshow(ventana_principal, vista_comparativa)
        
        # Esperar 1ms por una tecla
        k = cv2.waitKey(1) & 0xFF
        
        # --- 4. Manejo de Teclas ---
        
        # 'q' = Salir (Quit)
        if k == ord('q'):
            print("Saliendo del programa...")
            break # Rompe el bucle interactivo
        
        # 's' = Guardar/Set
        if k == ord('s'):
            print(f"  -> Umbral {umbral_valor} guardado para {filename}")
            resultados.append([filename, umbral_valor])
            break # Rompe el bucle interactivo y pasa a la siguiente imagen

        # Manejar si el usuario cierra la ventana con la 'X'
        try:
            if cv2.getWindowProperty(ventana_principal, cv2.WND_PROP_VISIBLE) < 1:
                k = ord('q') # Tratar como si hubiera presionado 'q'
                break
        except cv2.error:
            # Ocurre si la ventana se cierra abruptamente
            k = ord('q')
            break
            
    # Si la tecla 'q' fue presionada, salir también del bucle principal
    if k == ord('q'):
        break

# --- 5. Limpiar ventanas de OpenCV ---
cv2.destroyAllWindows()

# --- 6. Guardar resultados en CSV ---
if resultados:
    archivo_csv = 'patentes.csv'
    print(f"\nGuardando {len(resultados)} resultado(s) en '{archivo_csv}'...")
    
    try:
        with open(archivo_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Escribir la cabecera
            writer.writerow(['Imagen', 'Umbral'])
            # Escribir todos los resultados
            writer.writerows(resultados)
        print("¡Guardado completado!")
    except IOError:
        print(f"ERROR: No se pudo escribir en el archivo '{archivo_csv}'.")
else:
    print("\nNo se guardó ningún resultado (proceso finalizado o 's' no fue presionado).")