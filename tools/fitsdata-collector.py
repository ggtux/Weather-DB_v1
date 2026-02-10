import os
import json
from datetime import datetime, timedelta
from glob import glob
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body
import numpy as np
from PIL import Image

# Konfiguration
SENSOR_JSON = '/home/geo/Arbeitsplatz/Weatherdata/Sensordaten/sensor-readings.json'
IMAGE_DIR = '/home/geo/Arbeitsplatz/Weatherdata/image'
FITS_DIR = '/home/geo/Arbeitsplatz/Weatherdata/fits'

# Hilfsfunktion: Timestamp aus Bilddateiname extrahieren
def extract_timestamp_from_filename(filename):
    # Beispiel: 20260203_075601.jpg
    base = os.path.basename(filename)
    try:
        dt = datetime.strptime(base[:15], '%Y%m%d_%H%M%S')
        return dt
    except ValueError:
        return None

# Hilfsfunktion: Nächstgelegenen Sensordatensatz zum Bild finden
def find_matching_sensor_data(image_time, sensor_data, max_diff=timedelta(minutes=10)):
    best = None
    best_diff = max_diff
    for entry in sensor_data:
        try:
            t = datetime.fromisoformat(entry['timestamp'])
        except Exception:
            continue
        diff = abs(image_time - t)
        if diff < best_diff:
            best = entry
            best_diff = diff
    return best


# Standortdaten
LAT = 51.4798
LON = 13.7319
HEIGHT = 95
LOCATION = EarthLocation(lat=LAT, lon=LON, height=HEIGHT)

###############################
# Theoretische Sternanzahl im ROI (Bortle 4)
# ROI: Kreis mit 40° Durchmesser (20° Radius) um Zenit
# Bortle 4: ca. 8.500 Sterne bis 6.1 mag am Himmel
# Fläche ROI: A = pi * r^2 = pi * 20^2 ≈ 1.257 Quadratgrad
# Sterne pro Quadratgrad: 8.500 / 41.253 ≈ 0.21
# Erwartete Sterne im ROI: 0.21 * 1.257 ≈ 264
THEORETICAL_STARS_BORTLE4_ROI = 264

# Hilfsfunktionen für FITS-Felder
def calc_moonphase(obs_time):
    # Näherung: 0=Neumond, 50=Halb, 100=Vollmond
    # Quelle: https://stackoverflow.com/a/49114508
    y = obs_time.year
    m = obs_time.month
    d = obs_time.day
    if m < 3:
        y -= 1
        m += 12
    k1 = int(365.25 * (y + 4712))
    k2 = int(30.6 * (m + 1))
    k3 = int(((y / 100) + 49) * 0.75) - 38
    jd = k1 + k2 + d + 59  # Julian day
    jd -= k3
    ip = (jd - 2451550.1) / 29.530588853
    ip -= int(ip)
    phase = round(ip * 100)
    return phase

def calc_moon_zenit(obs_time):
    t = Time(obs_time)
    altaz = AltAz(obstime=t, location=LOCATION)
    moon = get_body('moon', t, location=LOCATION)
    alt = moon.transform_to(altaz).alt.deg
    return 90 - alt  # Abstand zum Zenit

def calc_sky_threshold(img_arr):
    # Dummy: Median als Schwellwert (später Otsu oder anderes Verfahren)
    return float(np.median(img_arr))

def get_roi_mask(shape, center=None, radius_deg=40, fov_deg=180):
    # Annahme: Fisheye, Bildmitte = Zenit, fov_deg = Bildfeld (z.B. 180°)
    h, w = shape
    if center is None:
        center = (w // 2, h // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    max_r = min(center[0], center[1])
    roi_r = max_r * (radius_deg / (fov_deg / 2))
    mask = r <= roi_r
    return mask

def calc_roi_stats(img_arr):
    mask = get_roi_mask(img_arr.shape)
    roi = img_arr[mask]
    median = float(np.median(roi))
    mean = float(np.mean(roi))
    # Dummy: Anzahl Sterne = Pixel > (Median + 3*Std)
    threshold = median + 3 * np.std(roi)
    n_stars = int(np.sum(roi > threshold))
    return median, mean, n_stars, roi

# FITS-Header-Keys (jetzt ohne Werte)
FITS_HEADER_KEYS = [
    ('MONDPHAS', 'Mondphase (0,1,2,3,4)'),
    ('MONDZEN', 'Mondabstand zum Zenit (deg)'),
    ('SKYTHRES', 'Schwellwert Himmelshintergrund'),
    ('ROI_MED', 'Median ROI (40deg um Zenit)'),
    ('ROI_MEAN', 'Mittelwert ROI (40deg um Zenit)'),
    ('ROISTAR', 'Anzahl Sterne ROI'),
]

# Hauptfunktion
def main():
    # Sensordaten laden
    with open(SENSOR_JSON, 'r') as f:
        sensor_data = json.load(f)
    # Alle Bilddateien finden
    image_files = sorted(glob(os.path.join(IMAGE_DIR, '*.jpg')))
    for img_path in image_files:
        img_time = extract_timestamp_from_filename(img_path)
        if img_time is None:
            print(f'Überspringe (kein gültiger Timestamp): {os.path.basename(img_path)}')
            continue
        sensor = find_matching_sensor_data(img_time, sensor_data)
        if not sensor:
            print(f'Kein Sensordatensatz für {img_path}')
            continue
        # Bilddaten laden
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img)
        # FITS-Header vorbereiten und Werte berechnen
        hdr = fits.Header()
        obs_time = img_time
        # Zeitangabe in hh:mm
        obs_time_hm = obs_time.strftime('%H:%M')
        hdr['OBSHM'] = (obs_time_hm, 'Beobachtungszeit (hh:mm)')
        # 1. Mondphase (nun als Prozentwert)
        mondphase = calc_moonphase(obs_time)
        # 2. Mondabstand zum Zenit
        mondzen = round(calc_moon_zenit(obs_time), 1)
        # 3. Schwellwert Himmelshintergrund
        sky_thresh = round(calc_sky_threshold(img_arr), 1)
        # 4-6. ROI-Statistiken
        roi_median, roi_mean, roi_stars, roi_pixels = calc_roi_stats(img_arr)
        roi_median = round(roi_median, 1)
        roi_mean = round(roi_mean, 1)
        # Header setzen
        hdr['MONDPHAS'] = (mondphase, 'Mondphase (%)')
        hdr['MONDZEN'] = (mondzen, 'Mondabstand zum Zenit (deg)')
        hdr['SKYTHRES'] = (sky_thresh, 'Schwellwert Himmelshintergrund')
        hdr['ROI_MED'] = (roi_median, 'Median ROI (40deg um Zenit)')
        hdr['ROI_MEAN'] = (roi_mean, 'Mittelwert ROI (40deg um Zenit)')
        hdr['ROISTAR'] = (roi_stars, 'Anzahl Sterne ROI')

        # Theoretische Werte nur bei fast wolkenlosem Himmel (WBG < 30%)
        wbg = None
        for k, v in sensor.items():
            if k.lower() in ['wbg', 'wolkendecke', 'cloudcover']:
                try:
                    wbg = float(v)
                except Exception:
                    pass
        if wbg is not None and wbg < 30:
            # Theoretische Werte berechnen
            n_theo = THEORETICAL_STARS_BORTLE4_ROI
            n_pix = len(roi_pixels)
            # Annahme: 264 Sterne mit Wert 255, Rest wie Hintergrund
            theo_median = round(float(np.median(np.concatenate([roi_pixels, np.full(n_theo, 255)]))), 1)
            theo_mean = round(float((np.sum(roi_pixels) + n_theo * 255) / (n_pix + n_theo)), 1)
            hdr['ROI_TMED'] = (theo_median, 'Theor. Median ROI (Bortle4, wolkenlos)')
            hdr['ROI_TMEA'] = (theo_mean, 'Theor. Mittelwert ROI (Bortle4, wolkenlos)')
            # Prozentanteil der tatsächlichen zu theoretischen Sternanzahl
            if n_theo > 0:
                star_percent = round(100.0 * roi_stars / n_theo, 1)
                hdr['ROI_SPC'] = (star_percent, 'Prozent reale/ideale Sterne im ROI')

        # Sensordaten in Header
        for k, v in sensor.items():
            if k == 'timestamp':
                continue
            try:
                # Runde Werte auf 1 Dezimalstelle (z.B. TOBJ, Temperatur, WBG)
                if isinstance(v, (int, float)) and k.lower() in ['tobj', 'temp', 'temperatur', 'tempa', 'tempb', 'wbg', 'wolkendecke', 'cloudcover']:
                    v = round(float(v), 1)
                hdr[k.upper()[:8]] = v
            except Exception:
                pass
        # FITS-Dateiname
        fits_name = os.path.splitext(os.path.basename(img_path))[0] + '.fits'
        fits_path = os.path.join(FITS_DIR, fits_name)
        # FITS schreiben
        hdu = fits.PrimaryHDU(img_arr, header=hdr)
        hdu.writeto(fits_path, overwrite=True)
        print(f'FITS gespeichert: {fits_path}')

if __name__ == '__main__':
    main()
