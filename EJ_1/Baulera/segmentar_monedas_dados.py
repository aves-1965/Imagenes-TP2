#!/usr/bin/env python3
"""
segmentar_monedas_dados.py

Programa en Python + OpenCV que procesa la imagen '/mnt/data/Monedas.png' y realiza:
 A) Segmentación automática de monedas y dados.
 B) Clasificación y conteo de distintos tipos de monedas.
 C) Detección del número visible en cada dado (carilla superior) y conteo.
 D) Informe final con parámetros aplicados en cada etapa.

El script guarda imágenes intermedias en './output/' y genera 'report.txt' y 'report.json'.

Requisitos: python3, opencv-python, numpy, matplotlib
"""

import os
import cv2
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Ruta del archivo (proporcionada en el contenedor)
# IMG_PATH = '/mnt/data/Monedas.png'
IMG_PATH = 'Monedas.png'
OUT_DIR = './output'
os.makedirs(OUT_DIR, exist_ok=True)

# Parámetros (guardados en el reporte). Ajustables si se necesita mayor robustez.
PARAMS = {
    'resize_max_width': 1600,
    'gaussian_ksize': (7,7),
    'gaussian_sigma': 0,
    'canny_thresh1': 60,
    'canny_thresh2': 180,
    'morph_kernel_size': 7,
    'min_object_area': 800,            # objetos pequeños filtrados
    'coin_circularity_thresh': 0.6,    # circularity > thresh -> moneda
    'kmeans_coin_classes': 3,          # cluster por tamaños de moneda
    'blob_min_area_pip': 20,           # pips mínimos
    'blob_max_area_pip': 900,          # pips máximos
}

###########################
# Utilidades
###########################

def save_img(path, img, cmap=None):
    cv2.imwrite(path, img)


def show_and_save(fig_path, img_bgr, title=None):
    # guarda también una versión con matplotlib para visualizar en entornos no-GUI
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    if title:
        plt.title(title)
#   plt.axis('off')
    plt.imshow(img_rgb)
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()

###########################
# Paso 0: cargar y preprocesar
###########################

def load_image(path, max_width=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    h,w = img.shape[:2]
    if max_width and w>max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

###########################
# Paso 1: conversión y suavizado
###########################

def to_gray_and_blur(img, params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, params['gaussian_ksize'], params['gaussian_sigma'])
    return gray, gauss

###########################
# Paso 2: segmentación inicial (fondo vs objetos)
###########################

def segment_by_threshold(gauss):
    # Otsu + morphological closing to get solid object blobs
    _, th = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_inv = 255 - th  # queremos objetos claros (monedas y dados) sean blancos
    return th_inv

###########################
# Paso 3: limpieza morfológica y extracción de contornos
###########################

def morphological_cleanup(bin_img, params):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['morph_kernel_size'], params['morph_kernel_size']))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k, iterations=1)
    return opened


def extract_contours(clean_bin, params):
    contours, _ = cv2.findContours(clean_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter by area
    filtered = [c for c in contours if cv2.contourArea(c) >= params['min_object_area']]
    return filtered

###########################
# Clasificación moneda vs dado
###########################

def contour_circularity(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim==0:
        return 0
    circ = 4*np.pi*area/(perim*perim)
    return circ

def separate_coins_and_dice(contours, params):
    coins = []
    dices = []
    for c in contours:
        circ = contour_circularity(c)
        if circ >= params['coin_circularity_thresh']:
            coins.append((c, circ))
        else:
            dices.append((c, circ))
    return coins, dices

###########################
# Clasificación de monedas por tamaño (k-means sobre diámetro equivalente)
###########################

def coin_size_features(coins):
    feats = []
    for c,_ in coins:
        area = cv2.contourArea(c)
        eq_diam = np.sqrt(4*area/np.pi)
        feats.append(eq_diam)
    return np.array(feats, dtype=np.float32).reshape(-1,1)

def cluster_coin_sizes(diameters, k):
    # cv2.kmeans requires float32
    if len(diameters)==0:
        return np.array([]), np.array([])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(diameters.astype(np.float32), k, None, criteria, 10, flags)
    labels = labels.flatten()
    centers = centers.flatten()
    # sort centers and remap labels so label 0 = smallest coin, etc.
    order = np.argsort(centers)
    new_centers = centers[order]
    remap = np.zeros_like(order)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    remapped_labels = remap[labels]
    return remapped_labels, new_centers

###########################
# Detección de puntos en dados (pips)
###########################

def detect_pips_in_die(img_gray, die_contour, params):
    # crop bounding box with slight padding
    x,y,w,h = cv2.boundingRect(die_contour)
    pad = int(0.08*max(w,h))
    x0 = max(0, x-pad); y0 = max(0, y-pad); x1 = min(img_gray.shape[1], x+w+pad); y1 = min(img_gray.shape[0], y+h+pad)
    crop = img_gray[y0:y1, x0:x1]

    # enhance contrast locally and threshold
    blur = cv2.GaussianBlur(crop, (7,7), 0)
    # adaptive threshold to isolate dark pips on bright dice
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 3)
    # remove small noise and fill
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    # find blobs
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pips = []
    for c in cnts:
        A = cv2.contourArea(c)
        if A < params['blob_min_area_pip'] or A > params['blob_max_area_pip']:
            continue
        # circularity check
        per = cv2.arcLength(c, True)
        circ = 4*np.pi*A/(per*per) if per>0 else 0
        if circ < 0.3:  # pips are roughly circular
            continue
        # compute center in original image coords
        M = cv2.moments(c)
        if M['m00']==0: continue
        cx = int(M['m10']/M['m00']) + x0
        cy = int(M['m01']/M['m00']) + y0
        pips.append(((cx,cy), A, circ))

    # for robustness, take number of pips as count of accepted blobs
    return pips, (x0,y0,x1,y1), th

###########################
# Report and visualizations
###########################

def visualize_detections(img_color, coins, coin_labels, coin_centers, dices, die_pip_results, params):
    out = img_color.copy()
    # draw coins with labels
    for idx,(c,circ) in enumerate(coins):
        M = cv2.moments(c)
        if M['m00']==0: continue
        cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
        color = (0,200,0)
        cv2.drawContours(out, [c], -1, color, 2)
        label = f"C{idx}:L{int(coin_labels[idx])}"
        cv2.putText(out, label, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # draw dice and pip counts
    for idx,(c,circ) in enumerate(dices):
        x,y,w,h = cv2.boundingRect(c)
        color = (200,0,0)
        cv2.rectangle(out, (x,y),(x+w,y+h), color, 2)
        pipinfo = die_pip_results.get(idx, {'pips':[]})
        n = len(pipinfo['pips'])
        cv2.putText(out, f"D{idx}:{n}", (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # draw pip centers
        for pip in pipinfo.get('pips',[]):
            (cx,cy), A, circ = pip
            cv2.circle(out, (cx,cy), 6, (0,0,255), -1)
    return out

###########################
# Programa principal
###########################

def main():
    report = {}
    report['params'] = PARAMS.copy()

    img = load_image(IMG_PATH, max_width=PARAMS['resize_max_width'])
    h,w = img.shape[:2]
    report['image_shape'] = (h,w)

    # Paso 1: grayscale + blur
    gray, gauss = to_gray_and_blur(img, PARAMS)
    cv2.imwrite(os.path.join(OUT_DIR, '01_gray.png'), gray)
    cv2.imwrite(os.path.join(OUT_DIR, '02_gauss.png'), gauss)
    report['steps'] = ['grayscale', 'gaussian_blur']

    # Paso 2: threshold (Otsu) -> invertir para objetos blancos
    th_inv = segment_by_threshold(gauss)
    cv2.imwrite(os.path.join(OUT_DIR, '03_otsu_inverted.png'), th_inv)
    report['steps'].append('otsu_threshold_inverted')

    # Paso 3: morfología
    clean = morphological_cleanup(th_inv, PARAMS)
    cv2.imwrite(os.path.join(OUT_DIR, '04_morph_clean.png'), clean)
    report['steps'].append('morphological_cleanup')

    # Extra: edges (Canny) para posibles usos
    edges = cv2.Canny(gauss, PARAMS['canny_thresh1'], PARAMS['canny_thresh2'])
    cv2.imwrite(os.path.join(OUT_DIR, '05_edges.png'), edges)
    report['steps'].append('canny_edges')

    # Paso 4: extraer contornos
    contours = extract_contours(clean, PARAMS)
    report['raw_contours_count'] = len(contours)

    # clasificar coin vs dice
    coins, dices = separate_coins_and_dice(contours, PARAMS)
    report['n_coins_initial'] = len(coins)
    report['n_dices_initial'] = len(dices)

"""
    # cluster monedas por tamaño
    coin_diams = coin_size_features(coins)
    coin_labels, coin_centers = cluster_coin_sizes(coin_diams, PARAMS['kmeans_coin_classes'])
    # if no clustering (0 coins) ensure arrays consistent
    if len(coin_labels)==0:
        coin_labels = np.array([])
        coin_centers = np.array([])
"""
    # detectar pips en dados
    die_pip_results = {}
    for i,(c,circ) in enumerate(dices):
        pips, bbox, th_local = detect_pips_in_die(gray, c, PARAMS)
        die_pip_results[i] = {'pips': pips, 'bbox': bbox}
        # save local threshold visualization
        x0,y0,x1,y1 = bbox
        viz = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        crop = th_local
        # normalize crop to 3-channels for saving easily
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        save_img(os.path.join(OUT_DIR, f'die_{i}_pips.png'), crop_rgb)

    # finalize report values for coins
    coin_summary = defaultdict(int)
    coin_classes_info = {}
    for i,(c,circ) in enumerate(coins):
        if len(coin_labels)>0:
            cls = int(coin_labels[i])
        else:
            cls = 0
        coin_summary[cls] += 1
    for cls_idx, center in enumerate(coin_centers.tolist() if len(coin_centers)>0 else []):
        coin_classes_info[int(cls_idx)] = {'centroid_equiv_diameter_px': float(center)}

    report['coin_summary'] = dict(coin_summary)
    report['coin_classes_info'] = coin_classes_info

    # prepare visualization image
    vis = visualize_detections(img, coins, coin_labels, coin_centers, dices, die_pip_results, PARAMS)
    show_and_save(os.path.join(OUT_DIR, '06_detections.png'), vis, title='Detections')
    cv2.imwrite(os.path.join(OUT_DIR, '06_detections_cv.png'), vis)

    # Print and save final report
    # Build more human-friendly entries
    final_objects = []
    for i,(c,circ) in enumerate(coins):
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00']) if M['m00'] else 0
        cy = int(M['m01']/M['m00']) if M['m00'] else 0
        area = cv2.contourArea(c)
        eq_diam = float(np.sqrt(4*area/np.pi))
        cls = int(coin_labels[i]) if len(coin_labels)>0 else 0
        final_objects.append({'type':'coin','id':i,'center':(cx,cy),'area':float(area),'equivalent_diameter_px':eq_diam,'class':cls})
    for i,(c,circ) in enumerate(dices):
        x,y,w,h = cv2.boundingRect(c)
        n_pips = len(die_pip_results.get(i, {}).get('pips', []))
        final_objects.append({'type':'die','id':i,'bbox':(int(x),int(y),int(w),int(h)),'pips':n_pips})

    report['final_objects'] = final_objects

    # Summaries
    report['counts'] = {
        'coins_total': sum(1 for o in final_objects if o['type']=='coin'),
        'dices_total': sum(1 for o in final_objects if o['type']=='die')
    }

    # Write textual report
    report_txt_lines = []
    report_txt_lines.append('REPORTE FINAL - Procesamiento de imagen')
    report_txt_lines.append('Ruta imagen: ' + IMG_PATH)
    report_txt_lines.append('\nPARAMETROS APLICADOS:')
    for k,v in PARAMS.items():
        report_txt_lines.append(f" - {k}: {v}")
    report_txt_lines.append('\nRESULTADOS:')
    report_txt_lines.append(f" - Imagen (h,w): {report['image_shape']}")
    report_txt_lines.append(f" - Objetos detectados (contornos filtrados): {report['raw_contours_count']}")
    report_txt_lines.append(f" - Monedas detectadas: {report['n_coins_initial']}")
    report_txt_lines.append(f" - Dados detectados: {report['n_dices_initial']}")

    report_txt_lines.append('\nDetalle monedas por clase (kmeans sobre diametros):')
    for cls, cnt in report['coin_summary'].items():
        center = report['coin_classes_info'].get(cls, {}).get('centroid_equiv_diameter_px', None)
        report_txt_lines.append(f" - Clase {cls}: cantidad={cnt}, diam_px_est={center}")

    report_txt_lines.append('\nDados y pips (cara superior detectada):')
    for o in final_objects:
        if o['type']=='die':
            report_txt_lines.append(f" - Dado id={o['id']}: pips_detectados={o['pips']}, bbox={o['bbox']}")

    report_txt_lines.append('\nARCHIVOS GENERADOS (en ./output):')
    for fname in sorted(os.listdir(OUT_DIR)):
        report_txt_lines.append(' - ' + fname)

    # save report text and json
    report_txt = '\n'.join(report_txt_lines)
    with open(os.path.join(OUT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_txt)
    with open(os.path.join(OUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(report_txt)
    print('\nReporte y visualizaciones guardadas en ./output/')

if __name__ == '__main__':
    main()
