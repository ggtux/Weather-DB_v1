"""
sky_temperature_model.py
========================
CloudWatcher Sky Temperature Correction Model — wiederverwendbares Modul.

Implementiert das Lunatico-Modell zur Korrektur der Infrarot-Himmelstemperatur
und schätzt den Wolkenbedeckungsgrad (WBG) daraus ab.

Verwendung im ALPACA Observing Conditions Driver::

    from sky_temperature_model import SkyTemperatureModel

    model = SkyTemperatureModel.from_json("calibration/sky_temp_calibration.json")

    ta   = 5.2    # Umgebungstemperatur [°C]  (OWM_TEMP)
    ts   = -12.3  # Rohe IR-Himmelstemperatur [°C]  (TOBJ)

    tsky = model.compute_Tsky(ta, ts)   # Korrigierte Himmelstemperatur [°C]
    wbg  = model.compute_WBG(ta, ts)    # Wolkenbedeckungsgrad [0..100 %]

Modell (Lunatico / CloudWatcher 2021):

    T67  = Kaltwitterungsfaktor(K2, K6, K7, Ta)
    Td   = (K1/100)*(Ta - K2/10) + (K3/100)*exp(K4/1000*Ta)^(K5/100) + T67
    Tsky = Ts - Td
    WBG  = a * Tsky + b      (lineare Abbildung, aus Kalibrierung)

Abhängigkeiten: numpy, json, os  (kein streamlit, scipy, astropy)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Union

import numpy as np

# Typ-Alias für skalare oder Array-Eingaben
Numeric = Union[float, np.ndarray]

# Standardpfad zur Kalibrierungsdatei (relativ zu diesem Modul)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(_MODULE_DIR), "calibration", "sky_temp_calibration.json"
)


# ---------------------------------------------------------------------------
# Modellformeln (modulare Hilfsfunktionen, auch direkt importierbar)
# ---------------------------------------------------------------------------

def compute_T67(K2: float, K6: float, K7: float, Ta: Numeric) -> Numeric:
    """
    T67 — Kaltwitterungsfaktor des CloudWatcher-Modells.

    Funktioniert mit skalaren Werten und numpy-Arrays.

    Parameters
    ----------
    K2, K6, K7 : float
        Modellkoeffizienten.
    Ta : float | np.ndarray
        Umgebungstemperatur [°C].

    Returns
    -------
    float | np.ndarray
        T67-Korrekturterm [°C].
    """
    scalar = np.ndim(Ta) == 0
    Ta = np.atleast_1d(np.asarray(Ta, dtype=float))

    x      = K2 / 10.0 - Ta
    abs_x  = np.abs(x)
    sgn_K6 = np.sign(K6)
    sgn_x  = np.sign(x)
    safe_x = np.where(abs_x < 1e-10, 1e-10, abs_x)

    branch1 = sgn_K6 * sgn_x * abs_x                               # |x| < 1
    branch2 = (K6 / 10.0) * sgn_x * np.log10(safe_x) + K7 / 100.0 # |x| >= 1
    result  = np.where(abs_x < 1.0, branch1, branch2)

    return float(result[0]) if scalar else result


def compute_Td(K: list[float], Ta: Numeric) -> Numeric:
    """
    Korrekturterm Td des CloudWatcher-Modells.

    Parameters
    ----------
    K : list[float]
        Koeffizientenliste [K1, K2, K3, K4, K5, K6, K7].
    Ta : float | np.ndarray
        Umgebungstemperatur [°C].

    Returns
    -------
    float | np.ndarray
        Korrekturterm Td [°C].
    """
    K1, K2, K3, K4, K5, K6, K7 = K
    T67      = compute_T67(K2, K6, K7, Ta)
    exp_arg  = np.clip(K4 / 1000.0 * np.asarray(Ta, dtype=float), -50.0, 50.0)
    exp_term = np.power(np.maximum(np.exp(exp_arg), 1e-10), K5 / 100.0)
    return (K1 / 100.0) * (np.asarray(Ta, dtype=float) - K2 / 10.0) \
           + (K3 / 100.0) * exp_term + T67


def compute_Tsky_raw(K: list[float], Ta: Numeric, Ts: Numeric) -> Numeric:
    """
    Korrigierte Himmelstemperatur Tsky = Ts − Td.

    Parameters
    ----------
    K : list[float]
        Koeffizientenliste [K1, K2, K3, K4, K5, K6, K7].
    Ta : float | np.ndarray
        Umgebungstemperatur [°C].
    Ts : float | np.ndarray
        Rohe IR-Himmelstemperatur [°C] (TOBJ).

    Returns
    -------
    float | np.ndarray
        Korrigierte Himmelstemperatur Tsky [°C].
    """
    return np.asarray(Ts, dtype=float) - compute_Td(K, Ta)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------

class SkyTemperatureModel:
    """
    CloudWatcher Sky Temperature Correction Model mit optionaler
    T_a-abhängiger Parameterinterpolation.

    Attribute
    ---------
    K_global : np.ndarray, shape (7,)
        Globaler Koeffizientensatz [K1..K7].
    a : float
        Steigung der linearen WBG-Abbildung (WBG = a·Tsky + b).
    b : float
        Achsenabschnitt der linearen WBG-Abbildung.
    poly_coeffs : dict | None
        Polynomkoeffizienten {„K1": array, …} für K_i(T_a).
        None wenn keine T_a-abhängige Kalibrierung vorhanden.
    poly_degree : int
        Grad der Polynomfunktion K_i(T_a).
    ta_range : tuple[float, float]
        Gültiger T_a-Bereich der Kalibrierung [min, max] [°C].
    """

    def __init__(
        self,
        K_global: list[float],
        a: float = 1.0,
        b: float = 0.0,
        poly_coeffs: dict | None = None,
        poly_degree: int = 1,
        ta_range: tuple[float, float] = (-50.0, 50.0),
        metadata: dict | None = None,
    ):
        self.K_global    = np.asarray(K_global, dtype=float)
        self.a           = float(a)
        self.b           = float(b)
        self.poly_coeffs = poly_coeffs          # {„K1": [c0, c1, ...], ...} oder None
        self.poly_degree = int(poly_degree)
        self.ta_range    = (float(ta_range[0]), float(ta_range[1]))
        self.metadata    = metadata or {}

    # ------------------------------------------------------------------
    # K-Werte für gegebenes T_a
    # ------------------------------------------------------------------

    def get_K_for_Ta(self, Ta: float) -> np.ndarray:
        """
        Gibt die interpolierten K1..K7 für eine bestimmte Umgebungstemperatur zurück.

        Sind keine Polynomkoeffizienten vorhanden, wird der globale K-Satz
        zurückgegeben. T_a wird auf den kalibrierten Bereich geclippt.

        Parameters
        ----------
        Ta : float
            Umgebungstemperatur [°C].

        Returns
        -------
        np.ndarray, shape (7,)
            Koeffizientensatz K1..K7.
        """
        if self.poly_coeffs is None:
            return self.K_global.copy()

        Ta_clipped = np.clip(float(Ta), self.ta_range[0], self.ta_range[1])
        return np.array([
            np.polyval(self.poly_coeffs[f"K{i+1}"], Ta_clipped)
            for i in range(7)
        ])

    # ------------------------------------------------------------------
    # Kernberechnungen
    # ------------------------------------------------------------------

    def compute_Tsky(self, Ta: Numeric, Ts: Numeric) -> Numeric:
        """
        Berechnet die korrigierte Himmelstemperatur Tsky.

        Verwendet T_a-abhängige K-Werte, falls eine Polynom-Kalibrierung
        vorhanden ist. Bei Array-Eingabe wird pro Datenpunkt interpoliert.

        Parameters
        ----------
        Ta : float | np.ndarray
            Umgebungstemperatur [°C].
        Ts : float | np.ndarray
            Rohe IR-Himmelstemperatur [°C].

        Returns
        -------
        float | np.ndarray
            Korrigierte Himmelstemperatur Tsky [°C].
        """
        scalar = np.ndim(Ta) == 0
        Ta = np.atleast_1d(np.asarray(Ta, dtype=float))
        Ts = np.atleast_1d(np.asarray(Ts, dtype=float))

        if self.poly_coeffs is None:
            # Globale K's: vektorisiert
            result = compute_Tsky_raw(self.K_global, Ta, Ts)
        else:
            # Pro Datenpunkt interpolierte K's
            result = np.array([
                compute_Tsky_raw(self.get_K_for_Ta(ta_i), ta_i, ts_i)
                for ta_i, ts_i in zip(Ta, Ts)
            ])

        return float(result[0]) if scalar else result

    def compute_WBG(self, Ta: Numeric, Ts: Numeric) -> Numeric:
        """
        Schätzt den Wolkenbedeckungsgrad (WBG) aus IR- und Umgebungstemperatur.

        WBG = a · Tsky + b  (aus Kalibrierung), geclippt auf [0, 100] %.

        Parameters
        ----------
        Ta : float | np.ndarray
            Umgebungstemperatur [°C].
        Ts : float | np.ndarray
            Rohe IR-Himmelstemperatur [°C].

        Returns
        -------
        float | np.ndarray
            Geschätzter Wolkenbedeckungsgrad [0..100 %].
        """
        Tsky = self.compute_Tsky(Ta, Ts)
        return np.clip(self.a * Tsky + self.b, 0.0, 100.0)

    def sky_condition(self, Ta: float, Ts: float) -> str:
        """
        Gibt eine qualitative Himmelsbeurteilung zurück.

        Parameters
        ----------
        Ta : float
            Umgebungstemperatur [°C].
        Ts : float
            Rohe IR-Himmelstemperatur [°C].

        Returns
        -------
        str
            „clear", „partly cloudy" oder „overcast".
        """
        wbg = float(self.compute_WBG(Ta, Ts))
        if wbg < 20.0:
            return "clear"
        elif wbg < 60.0:
            return "partly cloudy"
        else:
            return "overcast"

    # ------------------------------------------------------------------
    # Persistenz (JSON)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialisiert das Modell als Dictionary (JSON-kompatibel)."""
        d = {
            "metadata":    self.metadata,
            "K_global":    self.K_global.tolist(),
            "a":           self.a,
            "b":           self.b,
            "poly_degree": self.poly_degree,
            "ta_range":    list(self.ta_range),
            "poly_coeffs": (
                {k: v.tolist() for k, v in self.poly_coeffs.items()}
                if self.poly_coeffs is not None else None
            ),
        }
        return d

    def to_json(self, path: str) -> None:
        """
        Speichert die Kalibrierung als JSON-Datei.

        Parameters
        ----------
        path : str
            Zieldateipfad (wird ggf. angelegt).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SkyTemperatureModel":
        """Stellt ein Modell aus einem Dictionary wieder her."""
        poly_coeffs = None
        if d.get("poly_coeffs") is not None:
            poly_coeffs = {k: np.array(v) for k, v in d["poly_coeffs"].items()}
        return cls(
            K_global    = d["K_global"],
            a           = d.get("a", 1.0),
            b           = d.get("b", 0.0),
            poly_coeffs = poly_coeffs,
            poly_degree = d.get("poly_degree", 1),
            ta_range    = tuple(d.get("ta_range", (-50.0, 50.0))),
            metadata    = d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "SkyTemperatureModel":
        """
        Lädt eine Kalibrierung aus einer JSON-Datei.

        Parameters
        ----------
        path : str
            Pfad zur JSON-Kalibrierungsdatei.

        Raises
        ------
        FileNotFoundError
            Wenn die Datei nicht existiert.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Kalibrierungsdatei nicht gefunden: {path}\n"
                "Bitte zuerst das Sky Temperature Calibration Tool ausführen "
                "und eine Kalibrierung speichern."
            )
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def default(cls) -> "SkyTemperatureModel":
        """
        Gibt ein Modell mit CloudWatcher-Standardwerten zurück
        (K1=100, K2..K7=0 → Tsky = Ts − Ta).
        """
        return cls(
            K_global=[100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            a=-1.0,
            b=20.0,
            metadata={"source": "CloudWatcher defaults (uncalibrated)"},
        )

    # ------------------------------------------------------------------
    # Darstellung
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        K = self.K_global
        pc = "vorhanden" if self.poly_coeffs else "nicht vorhanden"
        return (
            f"SkyTemperatureModel("
            f"K=[{', '.join(f'{k:.2f}' for k in K)}], "
            f"a={self.a:.3f}, b={self.b:.2f}, "
            f"Polynom-Kalibrierung={pc}, "
            f"T_a-Bereich={self.ta_range})"
        )

    def summary(self) -> str:
        """Gibt eine mehrzeilige Zusammenfassung des Modells zurück."""
        lines = [
            "=" * 55,
            "  CloudWatcher Sky Temperature Model — Kalibrierung",
            "=" * 55,
        ]
        if self.metadata:
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
            lines.append("-" * 55)
        for i, ki in enumerate(self.K_global):
            lines.append(f"  K{i+1} (global) = {ki:.4f}")
        lines.append(f"  WBG = {self.a:.4f} · Tsky + {self.b:.4f}")
        lines.append(f"  T_a-Bereich: {self.ta_range[0]:.1f} … {self.ta_range[1]:.1f} °C")
        if self.poly_coeffs:
            lines.append(f"  Polynomgrad: {self.poly_degree}")
            lines.append("  Polynomkoeffizienten K_i(T_a):")
            for i in range(7):
                kname = f"K{i+1}"
                c = self.poly_coeffs[kname]
                lines.append(f"    {kname}: {np.array2string(c, precision=4)}")
        lines.append("=" * 55)
        return "\n".join(lines)
