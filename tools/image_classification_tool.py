import streamlit as st
import sqlite3
from PIL import Image
import numpy as np
import os

# --- Konfiguration ---
DB_PATH = "../weather.db"  # Passe ggf. an
IMAGE_DIR = "../Weatherdata/image/"
ROI_IMAGE_DIR = "../Weatherdata/image/roi/"

# --- Hilfsfunktionen ---
def get_image_list(image_dir):
    return [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

def load_metadata(image_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, wbg, moon_phase, moon_zenith_angle, classification, comment FROM image_metadata WHERE image_path=?", (image_name,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(zip(["timestamp", "wbg", "moon_phase", "moon_zenith_angle", "classification", "comment"], row))
    return None

def save_classification(image_name, classification, comment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE image_metadata SET classification=?, comment=? WHERE image_path=?", (classification, comment, image_name))
    conn.commit()
    conn.close()

# --- Streamlit UI ---
st.title("Interaktives Bildklassifizierungs-Tool (WBG/Mondphase)")

image_list = get_image_list(ROI_IMAGE_DIR)
if not image_list:
    st.warning("Keine Bilder im ROI-Ordner gefunden.")
    st.stop()

image_name = st.selectbox("Bild auswählen:", image_list)
image_path = os.path.join(ROI_IMAGE_DIR, image_name)


# Bild laden und in Graustufen umwandeln
image = Image.open(image_path).convert("L")
image_np = np.array(image)

brightness = 1.3
image_np_adj = np.clip(image_np * brightness, 0, 255).astype(np.uint8)



# 40° ROI als Kreis einzeichnen (zentral, Radius 676px)
import cv2
roi_image = cv2.cvtColor(image_np_adj, cv2.COLOR_GRAY2BGR)
height, width = roi_image.shape[:2]
center = (width // 2, height // 2)
roi_radius = 676
cv2.circle(roi_image, center, roi_radius, (0, 255, 0), 3)

# Schwellwert für Binarisierung (Standardwert: 26 für wolkenfreien Himmel)
st.subheader("Schwellwert für Binarisierung")
threshold = st.slider("Schwellwert (0=schwarz, 255=weiß)", min_value=0, max_value=255, value=26, step=1)

# Binarisiertes Bild erzeugen (und speichern)
binary_np = np.where(image_np_adj < threshold, 0, 255).astype(np.uint8)
binary_image = cv2.cvtColor(binary_np, cv2.COLOR_GRAY2BGR)
cv2.circle(binary_image, center, roi_radius, (0, 255, 0), 3)

# S/W-Bild dauerhaft speichern (bei jedem Aufruf mit aktuellem Schwellwert)
from PIL import Image as PILImage
sw_filename = os.path.splitext(image_name)[0] + f"_bw_{threshold}.png"
sw_path = os.path.join(ROI_IMAGE_DIR, sw_filename)
PILImage.fromarray(binary_np).save(sw_path)


# Anteil weißer Pixel innerhalb der ROI berechnen
y_indices, x_indices = np.ogrid[:height, :width]
mask = (x_indices - center[0])**2 + (y_indices - center[1])**2 <= roi_radius**2
roi_pixels = binary_np[mask]
white_pixels = np.count_nonzero(roi_pixels == 255)
total_pixels = roi_pixels.size
white_percent = 100 * white_pixels / total_pixels if total_pixels > 0 else 0

# Bild und WBG-Prozentanzeige nebeneinander (nur binarisiertes Bild)
col1, col2 = st.columns([3,1])
col1.image(binary_image, caption=f"ROI (40°) Binarisiertes Bild (Schwellwert: {threshold})", width='content')
col2.metric("Wolkenbedeckung (WBG)", f"{white_percent:.2f} %")

# Metadaten laden
meta = load_metadata(image_name)
if meta:
    st.write(f"**Zeitpunkt:** {meta['timestamp']}")
    st.write(f"**WBG (auto):** {meta['wbg']} %")
    st.write(f"**Mondphase:** {meta['moon_phase']} %")
    st.write(f"**Mondwinkel zum Zenit:** {meta['moon_zenith_angle']}°")
else:
    st.info("Keine Metadaten in DB gefunden.")

# Klassifizierung
st.subheader("Klassifizierung setzen")
moon_class = st.select_slider(
    "Mondphase [%]",
    options=[0,25,50,75,100],
    value=int(meta["moon_phase"]) if meta and meta["moon_phase"] is not None else 0
)
comment = st.text_input("Kommentar", value=meta["comment"] if meta and meta["comment"] else "")

if st.button("Speichern"):
    save_classification(image_name, round(white_percent,2), comment)
    st.success("Klassifizierung gespeichert!")

# Referenzwert für wolkenfreien Nachthimmel (ohne Mond):
# Schwellwert = 33 → 0,21% weiße Pixel (entspricht Sternenlicht)
# Werte darüber deuten auf Wolken oder Mondlicht hin.
