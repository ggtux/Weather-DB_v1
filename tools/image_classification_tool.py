
import streamlit as st
import os
import numpy as np
from astropy.io import fits
from PIL import Image

# --- Konfiguration ---
FITS_DIR = "../Weatherdata/fits/"

# --- Hilfsfunktionen ---
def get_fits_list(fits_dir):
    fits_files = [f for f in os.listdir(fits_dir) if f.lower().endswith(".fits")]
    # Sortiere numerisch nach Dateinamen (z.B. 20260203_075601.fits)
    def extract_num(fname):
        base = os.path.splitext(fname)[0]
        # Extrahiere Zahlen aus Dateiname
        nums = ''.join([c for c in base if c.isdigit()])
        return int(nums) if nums else 0
    return sorted(fits_files, key=extract_num)

def load_fits_data(fits_path):
    with fits.open(fits_path, mode='update') as hdul:
        img_data = hdul[0].data
        header = hdul[0].header
        return np.array(img_data), header, hdul

def save_threshold_to_fits(hdul, new_threshold):
    hdul[0].header['SKYTHRES'] = (float(new_threshold), 'Schwellwert Himmelshintergrund')
    hdul.flush()

# --- Streamlit UI ---

st.title("Interaktives FITS-Bildklassifizierungs-Tool (WBG/Mondphase)")

fits_list = get_fits_list(FITS_DIR)
if not fits_list:
    st.warning("Keine FITS-Dateien gefunden.")
    st.stop()


# Layout: Buttons und Dropdown nebeneinander
col_nav1, col_dropdown, col_nav2 = st.columns([1,3,1])
fits_idx = fits_list.index(fits_list[0]) if fits_list else 0
if 'fits_idx' not in st.session_state:
    st.session_state['fits_idx'] = fits_idx

if col_nav1.button('<< Rückwärts'):
    st.session_state['fits_idx'] = max(0, st.session_state['fits_idx'] - 1)
if col_nav2.button('Vorwärts >>'):
    st.session_state['fits_idx'] = min(len(fits_list)-1, st.session_state['fits_idx'] + 1)

fits_name = col_dropdown.selectbox("FITS-Datei auswählen:", fits_list, index=st.session_state['fits_idx'])
st.session_state['fits_idx'] = fits_list.index(fits_name)
fits_path = os.path.join(FITS_DIR, fits_name)


# FITS laden
img_data, header, hdul = load_fits_data(fits_path)

# Umschaltung RGB/SW
st.subheader("Bilddarstellung")
view_mode = st.radio("Darstellung wählen:", ["Graustufen", "RGB"])

import cv2
height, width = img_data.shape[:2]
center = (width // 2, height // 2)
roi_radius = int(min(center) * (40 / 90))  # 40° von 90° (Halbbild)

if view_mode == "Graustufen":
    image_np = img_data.astype(np.uint8)
    brightness = 1.3
    image_np_adj = np.clip(image_np * brightness, 0, 255).astype(np.uint8)
    roi_image = cv2.cvtColor(image_np_adj, cv2.COLOR_GRAY2BGR)
else:
    image_np = img_data.astype(np.uint8)
    image_rgb = np.stack([image_np]*3, axis=-1)
    roi_image = image_rgb.copy()
    image_np_adj = image_np.copy()  # Für Threshold-Berechnung

cv2.circle(roi_image, center, roi_radius, (0, 255, 0), 3)


# Schwellwert für Binarisierung (Startwert aus FITS-Header)
st.subheader("Schwellwert für Binarisierung")
fits_threshold = int(header.get('SKYTHRES', 26))
threshold = st.slider("Schwellwert (0=schwarz, 255=weiß)", min_value=0, max_value=255, value=fits_threshold, step=1)


# Binarisiertes Bild erzeugen
binary_np = np.where(image_np_adj < threshold, 0, 255).astype(np.uint8)
binary_image = cv2.cvtColor(binary_np, cv2.COLOR_GRAY2BGR)
cv2.circle(binary_image, center, roi_radius, (0, 255, 0), 3)



# Anteil weißer Pixel innerhalb der ROI berechnen
y_indices, x_indices = np.ogrid[:height, :width]
mask = (x_indices - center[0])**2 + (y_indices - center[1])**2 <= roi_radius**2
roi_pixels = binary_np[mask]
white_pixels = np.count_nonzero(roi_pixels == 255)
total_pixels = roi_pixels.size
white_percent = 100 * white_pixels / total_pixels if total_pixels > 0 else 0



# Bild und WBG-Prozentanzeige nebeneinander
col1, col2 = st.columns([3,1])
if view_mode == "Graustufen":
    col1.image(binary_image, caption=f"ROI (40°) Binarisiertes Bild (Schwellwert: {threshold})", width='content')
else:
    # RGB: zeige das ROI-Bild in Farbe
    col1.image(roi_image, caption=f"ROI (40°) RGB-Bild", channels="RGB", width='content')
col2.metric("Wolkenbedeckung (WBG)", f"{white_percent:.2f} %")

# FITS-Headerdaten tabellarisch anzeigen
st.subheader("FITS-Headerdaten")

# FITS-Header ab MONDPHAS dynamisch ausgeben, WBG ergänzen
fits_keys = list(header.keys())
if 'MONDPHAS' in fits_keys:
    idx = fits_keys.index('MONDPHAS')
    # Alle Header ab MONDPHAS
    fits_keys = fits_keys[idx:]
else:
    fits_keys = ['MONDPHAS','MONDZEN','SKYTHRES','ROI_MED','ROI_MEAN','ROI_STARS']
# WBG ergänzen, falls nicht vorhanden
if 'WBG' not in fits_keys:
    fits_keys.append('WBG')
fits_table = {k: header.get(k, '') for k in fits_keys}
st.table(fits_table)



# Werte speichern (Threshold, WBG, alle ab MONDPHAS)
if st.button("Werte in FITS speichern"):
    # Threshold speichern
    hdul[0].header['SKYTHRES'] = (float(threshold), 'Schwellwert Himmelshintergrund')
    # WBG speichern
    hdul[0].header['WBG'] = (float(white_percent), 'Wolkenbedeckung (%)')
    # Alle weiteren Werte ab MONDPHAS (falls editierbar, hier nur flush)
    hdul.flush()
    st.success("Werte im FITS-Header gespeichert!")
