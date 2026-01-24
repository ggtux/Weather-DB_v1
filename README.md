# Weather-DB v1

Eine Python-basierte Wetterdatenbank zur Integration von:
- Private Wetterstation (PWS) API-Daten
- AllSky Kamera Himmelsbildanalyse
- Wolkentemperaturmessungen und -berechnungen
- RG-11 Regensensor
- Astronomische Berechnungen (Mondphase, Position)
- Web-Interface zur Datenvisualisierung

## Features

- 🌤️ JSON-basierte API-Integration für Wetterstationsdaten
- 📷 Automatische Bildanalyse der AllSky Kamera (alle 5 Min)
- 🌡️ Berechnung von Himmelstemperatur (Td), Wolkentemperatur (Tsky) und Wolkenbedeckungsgrad (WBG)
- 🌙 Mondphase und Winkelabstand zum Zenit
- 🌧️ Integration des RG-11 Regensensors
- 🔭 ROI-Definition um den Zenit (30° Radius)
- 📊 Web-Interface zur Analyse und Darstellung
- 🎯 Optimierung der Berechnungskoeffizienten

**Standort:** 51,4798° N, 13,7319° E, 95m ü.N.N.

## Requirements

- Python 3.9 oder höher
- pip für Paketinstallation
- Optional: virtualenv oder conda

## Installation

### 1. Repository klonen

```bash
git clone https://github.com/yourusername/Weather-DB_v1.git
cd Weather-DB_v1
```

### 2. Virtual Environment erstellen (empfohlen)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 4. Konfiguration

```bash
cp config.yml.example config.yml
# Passe config.yml mit deinen API-Endpoints und Zugangsdaten an
```

## Verwendung

### Anwendung starten

```bash
python src/main.py
```
streamlit run tools/image_classification_tool.py

### Development Mode

```bash
# Mit Flask Web-Interface
export FLASK_ENV=development
python src/main.py
```

## Projektstruktur

```
Weather-DB_v1/
├── src/
│   ├── main.py              # Hauptprogramm
│   ├── api/                 # API-Integration Module
│   ├── image_analysis/      # Bildverarbeitung
│   ├── calculations/        # Wolken- & Astro-Berechnungen
│   ├── database/            # Datenbankmodelle
│   └── web/                 # Web-Interface
├── include/                 # Konfigurationen
├── data/                    # Lokale Daten & DB (nicht in Git)
├── tests/                   # Unit-Tests
├── .github/
│   ├── Pflichtenheft.txt    # Projektspezifikation
│   └── workflows/           # GitHub Actions CI/CD
├── requirements.txt         # Python-Abhängigkeiten
├── setup.py                 # Package-Setup
├── config.yml.example       # Beispiel-Konfiguration
├── .gitignore
└── README.md
```

## Berechnungsformel (Td)

```python
K1 = 30.0  # Startwerte für Optimierung
K2 = 125.0
K3 = 0.0
K4 = 0.0
K5 = 0.0
K6 = 0.0
K7 = 0.0

if abs((K2 / 10. - Tamb)) < 1:
    T67 = sign(K6) * sign(Tamb - K2 / 10.) * abs((K2 / 10. - Tamb))
else:
    T67 = K6 / 10. * sign(Tamb - K2 / 10.) * (log(abs((K2 / 10. - Tamb))) / log(10.) + K7 / 100.)

Td = (K1 / 100.) * (Tamb - K2 / 10.) + (K3 / 100.) * pow(exp(K4 / 1000. * Tamb), (K5 / 100.)) + T67
Tsky = Tobj - Td
```

## Tests ausführen

```bash
pytest tests/
# Mit Coverage
pytest --cov=src tests/
```

## Development

### Neue Features hinzufügen

1. Erstelle Module in `src/`
2. Füge Tests in `tests/` hinzu
3. Aktualisiere `requirements.txt` bei neuen Abhängigkeiten
4. Dokumentiere in README

### Code-Style

- Folge PEP 8
- Nutze Type Hints
- Dokumentiere Funktionen mit Docstrings

## API-Endpunkte (geplant)

- `GET /api/weather/current` - Aktuelle Wetterdaten
- `GET /api/sky/image` - Letztes AllSky Bild
- `GET /api/cloud/temperature` - Wolkentemperatur
- `GET /api/moon/phase` - Mondphase
- `GET /api/rain/status` - Regenstatus

## Lizenz

[Add your license here]

## Autor

Georg - Weather-DB Projekt
