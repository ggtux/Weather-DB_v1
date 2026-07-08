#!/usr/bin/env python3
"""
CloudWatcher Sky Temperature Calibration Tool
==============================================
Optimiert K1–K7 des CloudWatcher Sky Temperature Correction Model so, dass
der aus T_sky berechnete Bedeckungsgrad (WBG_model) den gemessenen WBG-Werten
bestmöglich entspricht (Minimierung des RMSE).

Modell:
    T67  = Kaltwitterungsfaktor (K2, K6, K7, Ta)
    Td   = (K1/100)*(Ta - K2/10) + (K3/100)*exp(K4/1000*Ta)^(K5/100) + T67
    Tsky = Ts - Td
    WBG_model = a*Tsky + b   (lineare Abbildung, analytisch optimal)

FITS-Header Mapping:
    TOBJ     → Ts  (rohe IR-Himmelstemperatur, °C)
    OWM_TEMP → Ta  (Umgebungstemperatur, °C)
    WBG      → Zielgröße (Wolkenbedeckungsgrad, 0–100 %)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from astropy.io import fits
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, pearsonr
from datetime import datetime

# Eigenes Modul (src/) importieren
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from sky_temperature_model import SkyTemperatureModel

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
FITS_REV_DIR   = os.path.join(PROJECT_ROOT, "Weatherdata", "fits", "fits-rev")
CALIBRATION_DIR = os.path.join(PROJECT_ROOT, "calibration")
CALIBRATION_JSON = os.path.join(CALIBRATION_DIR, "sky_temp_calibration.json")

# ---------------------------------------------------------------------------
# CloudWatcher-Modell
# ---------------------------------------------------------------------------

def compute_T67(K2, K6, K7, Ta):
    """T67 Kaltwitterungsfaktor (vektorisiert)."""
    x        = K2 / 10.0 - Ta
    abs_x    = np.abs(x)
    sgn_K6   = np.sign(K6)
    sgn_x    = np.sign(x)
    safe_x   = np.where(abs_x < 1e-10, 1e-10, abs_x)
    branch1  = sgn_K6 * sgn_x * abs_x                            # |x| < 1
    branch2  = (K6 / 10.0) * sgn_x * np.log10(safe_x) + K7 / 100.0  # |x| >= 1
    return np.where(abs_x < 1.0, branch1, branch2)


def compute_Tsky(K, Ta, Ts):
    """Berechnet korrigierte Himmelstemperatur T_sky = Ts - Td."""
    K1, K2, K3, K4, K5, K6, K7 = K
    T67      = compute_T67(K2, K6, K7, Ta)
    exp_arg  = np.clip(K4 / 1000.0 * Ta, -50.0, 50.0)
    exp_term = np.power(np.maximum(np.exp(exp_arg), 1e-10), K5 / 100.0)
    Td       = (K1 / 100.0) * (Ta - K2 / 10.0) + (K3 / 100.0) * exp_term + T67
    return Ts - Td


def linear_wbg_model(Tsky, WBG):
    """
    Analytisch optimale lineare Abbildung WBG = a*Tsky + b.
    Gibt (a, b, WBG_predicted, RMSE) zurück.
    """
    A = np.column_stack([Tsky, np.ones(len(Tsky))])
    coeffs, _, _, _ = np.linalg.lstsq(A, WBG, rcond=None)
    a, b = coeffs
    WBG_pred = a * Tsky + b
    rmse = np.sqrt(np.mean((WBG - WBG_pred) ** 2))
    return a, b, WBG_pred, rmse


def loss_rmse(K, Ta, Ts, WBG):
    """
    Zielfunktion: RMSE zwischen gemessenem WBG und WBG_model.
    WBG_model = a*Tsky + b  (a,b analytisch optimal für die aktuellen K's).
    """
    try:
        Tsky = compute_Tsky(K, Ta, Ts)
        if not np.all(np.isfinite(Tsky)):
            return 1e6
        _, _, _, rmse = linear_wbg_model(Tsky, WBG)
        return rmse if np.isfinite(rmse) else 1e6
    except Exception:
        return 1e6

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(fits_dir):
    records, skipped = [], []
    if not os.path.isdir(fits_dir):
        return pd.DataFrame(), [f"Verzeichnis nicht gefunden: {fits_dir}"]
    for fname in sorted(os.listdir(fits_dir)):
        if not fname.lower().endswith(".fits"):
            continue
        try:
            with fits.open(os.path.join(fits_dir, fname), mode="readonly") as hdul:
                h   = hdul[0].header
                ts  = h.get("TOBJ")
                ta  = h.get("OWM_TEMP")
                wbg = h.get("WBG")
                if ts is not None and ta is not None and wbg is not None:
                    records.append({
                        "Datei":   fname,
                        "Ts [°C]": float(ts),
                        "Ta [°C]": float(ta),
                        "WBG [%]": float(wbg),
                    })
                else:
                    skipped.append(fname)
        except Exception:
            skipped.append(fname)
    return pd.DataFrame(records), skipped

# ---------------------------------------------------------------------------
# Optimierung
# ---------------------------------------------------------------------------

def run_optimization(bounds, Ta, Ts, WBG, maxiter=800, popsize=20, progress_bar=None):
    gen_count = [0]

    def callback(xk, convergence):
        gen_count[0] += 1
        if progress_bar is not None:
            pct = min(gen_count[0] / maxiter, 0.99)
            progress_bar.progress(pct,
                text=f"Generation {gen_count[0]}  |  Konvergenz: {convergence:.5f}")
        return False

    result = differential_evolution(
        loss_rmse, bounds, args=(Ta, Ts, WBG),
        maxiter=maxiter, tol=1e-8,
        seed=42, popsize=popsize,
        callback=callback, polish=True,
    )
    if progress_bar is not None:
        progress_bar.progress(1.0, text="Optimierung abgeschlossen.")
    return result

# ---------------------------------------------------------------------------
# App-Layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sky Temp Calibration", layout="wide")
st.title("CloudWatcher Kalibrierung — K1 bis K7")
st.caption(
    "Minimierung des RMSE zwischen gemessenem WBG und berechnetem WBG_Modell = a·T_sky + b  "
    "| CloudWatcher Sky Temperature Correction Model (Lunatico 2021)"
)

# --- Daten laden ---
df, skipped = load_data(FITS_REV_DIR)
if df.empty:
    st.error(f"Keine verwertbaren FITS-Dateien in:\n`{FITS_REV_DIR}`")
    st.stop()

Ta  = df["Ta [°C]"].values
Ts  = df["Ts [°C]"].values
WBG = df["WBG [%]"].values

# Infoleiste
c1, c2, c3, c4 = st.columns(4)
c1.metric("Datenpunkte",   len(df))
c2.metric("T_a Bereich",   f"{Ta.min():.1f} … {Ta.max():.1f} °C")
c3.metric("T_s Bereich",   f"{Ts.min():.1f} … {Ts.max():.1f} °C")
c4.metric("WBG Bereich",   f"{WBG.min():.0f} … {WBG.max():.0f} %")
if skipped:
    st.warning(f"{len(skipped)} Dateien ohne TOBJ/OWM_TEMP/WBG übersprungen.")

# --- Datensatz & CSV-Export ---
with st.expander("Datensatz & Exploration", expanded=False):
    st.dataframe(df, use_container_width=True)
    csv_raw = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(
        "CSV herunterladen (für Excel/Calc)",
        data=csv_raw, file_name="sky_temp_rohdaten.csv", mime="text/csv",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc = axes[0].scatter(df["Ta [°C]"], df["Ts [°C]"],
                         c=df["WBG [%]"], cmap="RdYlGn_r",
                         alpha=0.85, edgecolors="none", s=45)
    plt.colorbar(sc, ax=axes[0], label="WBG [%]")
    axes[0].set(xlabel="T_a [°C]", ylabel="T_s [°C]", title="T_s vs T_a  (Farbe = WBG)")
    axes[0].grid(alpha=0.3)
    axes[1].hist(df["WBG [%]"], bins=20, color="steelblue", edgecolor="white", rwidth=0.85)
    axes[1].set(xlabel="WBG [%]", title="WBG-Verteilung")
    axes[2].hist(df["Ta [°C]"], bins=20, color="darkorange", edgecolor="white", rwidth=0.85)
    axes[2].set(xlabel="T_a [°C]", title="T_a-Verteilung")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---------------------------------------------------------------------------
# Sidebar: Optimierungsgrenzen & Parameter
# ---------------------------------------------------------------------------
st.sidebar.header("Optimierungseinstellungen")
DEFAULTS = [
    ("K1",  0,    200),
    ("K2", -200,  200),
    ("K3", -100,  100),
    ("K4", -20,   20 ),
    ("K5",  0,    200),
    ("K6", -100,  100),
    ("K7", -100,  100),
]
bounds = []
for name, lo_d, hi_d in DEFAULTS:
    lo, hi = st.sidebar.slider(name, -400, 400, (lo_d, hi_d), step=5)
    bounds.append((lo, hi))
st.sidebar.divider()
popsize = st.sidebar.slider("Populationsgröße", 10, 40, 20, step=5)
maxiter = st.sidebar.slider("Max. Generationen", 200, 2000, 800, step=100)

# ---------------------------------------------------------------------------
# 1. Globale Optimierung
# ---------------------------------------------------------------------------
st.header("1. Globaler K1–K7-Satz")
st.markdown(
    "Optimiert **einen** Satz K1–K7 über alle Daten. "
    "Für jede Kandidatenlösung K wird T_sky berechnet, dann analytisch "
    "die beste lineare Abbildung **WBG_Modell = a·T_sky + b** bestimmt "
    "und der RMSE (WBG_gemessen − WBG_Modell) minimiert."
)

if st.button("▶ Globale Optimierung starten", type="primary"):
    bar    = st.progress(0, text="Starte…")
    result = run_optimization(bounds, Ta, Ts, WBG,
                              maxiter=maxiter, popsize=popsize, progress_bar=bar)
    K_opt  = result.x
    Tsky   = compute_Tsky(K_opt, Ta, Ts)
    a, b, WBG_pred, rmse = linear_wbg_model(Tsky, WBG)
    rho, _ = spearmanr(Tsky, WBG)
    r, _   = pearsonr(Tsky, WBG)

    st.session_state["K_global"]  = K_opt
    st.session_state["ab_global"] = (a, b)

    # Metriken
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE [%WBG]",    f"{rmse:.2f}")
    m2.metric("Spearman ρ",     f"{rho:.4f}")
    m3.metric("Pearson r",      f"{r:.4f}")
    m4.metric("Generationen",   result.nit)
    m5.metric("Konvergiert",    "Ja" if result.success else "Nein (Limit)")

    # K-Tabelle
    k_df = pd.DataFrame({"K": [f"K{i+1}" for i in range(7)], "Wert": np.round(K_opt, 4)})
    st.caption(f"Lineare Abbildung: WBG = {a:.4f}·T_sky {b:+.4f}")
    st.table(k_df.set_index("K").T)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(Tsky, WBG, c=Ta, cmap="coolwarm",
                         alpha=0.85, edgecolors="none", s=45, label="Messung")
    axes[0].scatter(Tsky, WBG_pred, marker="x", c="black", s=30, alpha=0.5, label="WBG_Modell")
    plt.colorbar(sc, ax=axes[0], label="T_a [°C]")
    xs = np.linspace(Tsky.min(), Tsky.max(), 200)
    axes[0].plot(xs, a * xs + b, "k--", lw=1.5,
                 label=f"WBG = {a:.2f}·T_sky {b:+.2f}")
    axes[0].set(xlabel="T_sky [°C]", ylabel="WBG [%]",
                title=f"T_sky vs WBG  (RMSE = {rmse:.2f} %,  ρ = {rho:.3f})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    residuals = WBG - WBG_pred
    r_ta, _   = pearsonr(Ta, residuals)
    axes[1].scatter(Ta, residuals, alpha=0.75, edgecolors="none", s=45, color="steelblue")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="T_a [°C]", ylabel="Residuum WBG [%]",
                title=f"Residuen vs T_a  (Pearson r = {r_ta:.3f})")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if abs(r_ta) > 0.35:
        st.warning(
            f"Die Residuen korrelieren mit T_a (r = {r_ta:.3f}). "
            "Eine T_a-abhängige Kalibrierung (Abschnitt 2) könnte den RMSE weiter reduzieren."
        )
    else:
        st.success(
            f"Keine starke T_a-Abhängigkeit der Residuen (r = {r_ta:.3f}). "
            "Der globale K-Satz erscheint robust."
        )

    # Globale Kalibrierung als JSON speichern
    if st.button("Globale Kalibrierung speichern (JSON)"):
        model = SkyTemperatureModel(
            K_global=K_opt.tolist(),
            a=float(a), b=float(b),
            ta_range=(float(Ta.min()), float(Ta.max())),
            metadata={
                "created":    datetime.now().isoformat(timespec="seconds"),
                "source":     "global_optimization",
                "n_samples":  int(len(Ta)),
                "rmse":       round(float(rmse), 4),
                "spearman_r": round(float(rho), 4),
                "pearson_r":  round(float(r), 4),
            },
        )
        model.to_json(CALIBRATION_JSON)
        st.success(f"Gespeichert: `{CALIBRATION_JSON}`")

# ---------------------------------------------------------------------------
# 2. T_a-Bin-Analyse
# ---------------------------------------------------------------------------
st.header("2. T_a-abhängige K-Parameter")
st.markdown(
    "Der Datensatz wird in T_a-Quantilsbins aufgeteilt und K1–K7 für jeden Bin separat optimiert. "
    "Anschließend wird ein Polynom K_i(T_a) angepasst. "
    "Im Abschnitt **3** können Sie T_a eingeben und die zugehörigen K's abrufen."
)

if "K_global" not in st.session_state:
    st.info("Bitte zuerst die **globale Optimierung** (Abschnitt 1) starten.")
else:
    n_bins      = st.slider("Anzahl T_a-Bins (Quantile)", 2, min(6, max(2, len(df) // 12)), 3)
    min_pts     = st.number_input("Mindestpunkte pro Bin", 5, 30, 10, step=1)
    poly_degree = st.slider("Polynomgrad für K_i(T_a)-Fit", 1, min(3, n_bins - 1), 1)

    if st.button("▶ Bin-Analyse starten"):
        try:
            labels, edges = pd.qcut(df["Ta [°C]"], q=n_bins, labels=False, retbins=True)
        except ValueError:
            st.error("Zu wenige unterschiedliche T_a-Werte für die gewählte Bin-Anzahl.")
            st.stop()

        bin_results = []
        for b in range(n_bins):
            mask = (labels == b).values
            if mask.sum() < min_pts:
                st.warning(f"Bin {b+1}: nur {mask.sum()} Punkte — übersprungen.")
                continue
            ta_b, ts_b, wbg_b = Ta[mask], Ts[mask], WBG[mask]
            span = f"[{ta_b.min():.1f}, {ta_b.max():.1f}] °C  n={mask.sum()}"

            bar_b = st.progress(0, text=f"Bin {b+1}/{n_bins}: T_a ∈ {span}")
            res   = run_optimization(bounds, ta_b, ts_b, wbg_b,
                                     maxiter=500, popsize=15, progress_bar=bar_b)

            Tsky_b = compute_Tsky(res.x, ta_b, ts_b)
            a_b, b_b, WBG_pred_b, rmse_b = linear_wbg_model(Tsky_b, wbg_b)
            rho_b, _ = spearmanr(Tsky_b, wbg_b)

            bin_results.append({
                "Bin":     b + 1,
                "Ta_min":  round(ta_b.min(), 2),
                "Ta_mean": round(ta_b.mean(), 2),
                "Ta_max":  round(ta_b.max(), 2),
                "n":       int(mask.sum()),
                "RMSE":    round(rmse_b, 3),
                "ρ":       round(rho_b, 4),
                **{f"K{i+1}": round(res.x[i], 4) for i in range(7)},
            })

        if not bin_results:
            st.error("Kein Bin mit ausreichend Datenpunkten.")
            st.stop()

        bin_df = pd.DataFrame(bin_results)
        ta_means = bin_df["Ta_mean"].values

        # Polynomfit K_i(T_a) für jeden Parameter
        poly_coeffs = {}
        for i in range(7):
            kname = f"K{i+1}"
            poly_coeffs[kname] = np.polyfit(ta_means, bin_df[kname].values, poly_degree)

        st.session_state["bin_df"]      = bin_df
        st.session_state["poly_coeffs"] = poly_coeffs
        st.session_state["poly_degree"] = poly_degree

        # --- Ergebnistabelle ---
        st.subheader("K-Werte je T_a-Bin")
        st.dataframe(bin_df, use_container_width=True)

        csv_bin = bin_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        st.download_button(
            "CSV Bin-Ergebnisse herunterladen",
            data=csv_bin, file_name="sky_temp_bins.csv", mime="text/csv",
        )

        # --- Plots K_i vs T_a ---
        K_global = st.session_state["K_global"]
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        axes = axes.flatten()
        xs_plot = np.linspace(ta_means.min(), ta_means.max(), 200)

        for i in range(7):
            ax     = axes[i]
            kname  = f"K{i+1}"
            k_vals = bin_df[kname].values
            ax.scatter(ta_means, k_vals, s=80, zorder=5, color="steelblue", label="Bin-Optimum")
            ax.axhline(K_global[i], color="gray", ls="--", lw=1.5,
                       label=f"Global: {K_global[i]:.2f}")
            y_poly = np.polyval(poly_coeffs[kname], xs_plot)
            coeffs_str = "  ".join([f"{c:+.3g}" for c in poly_coeffs[kname]])
            ax.plot(xs_plot, y_poly, "r-", lw=1.8,
                    label=f"Poly{poly_degree}: [{coeffs_str}]")
            ax.set(xlabel="T_a [°C]", ylabel=kname, title=kname)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # RMSE & Spearman je Bin
        ax = axes[7]
        bar_w = max((ta_means.max() - ta_means.min()) / max(n_bins, 1) * 0.6, 0.3)
        ax.bar(ta_means, bin_df["RMSE"].values, width=bar_w, color="tomato",  alpha=0.8, label="RMSE [%]")
        ax.bar(ta_means, bin_df["ρ"].values,    width=bar_w, color="steelblue", alpha=0.5, label="Spearman ρ")
        ax.set(xlabel="T_a [°C]", title="Fitqualität je Bin")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        axes[8].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- T_a-Abhängigkeitstabelle ---
        st.subheader("Quantifizierung der T_a-Abhängigkeit")
        dep_rows = []
        for i in range(7):
            kname  = f"K{i+1}"
            k_vals = bin_df[kname].values
            span   = float(k_vals.max() - k_vals.min())
            r_ta   = float(pearsonr(ta_means, k_vals)[0]) if len(ta_means) >= 3 else float("nan")
            dep_rows.append({
                "Parameter":   kname,
                "Min":         round(k_vals.min(), 3),
                "Max":         round(k_vals.max(), 3),
                "Spanne":      round(span, 3),
                "r(K, T_a)":   round(r_ta, 3),
                "Abhängigkeit": (
                    "stark"  if abs(r_ta) > 0.70 else
                    "mittel" if abs(r_ta) > 0.40 else
                    "gering"
                ),
            })
        dep_df = pd.DataFrame(dep_rows)
        st.dataframe(dep_df, use_container_width=True)

        strong = [r["Parameter"] for r in dep_rows if r["Abhängigkeit"] == "stark"]
        if strong:
            st.warning(
                f"Starke T_a-Abhängigkeit bei: **{', '.join(strong)}**. "
                "Das Polynom K_i(T_a) sollte für diese Parameter verwendet werden."
            )
        else:
            st.success(
                "Kein K-Parameter zeigt eine starke T_a-Abhängigkeit. "
                "Der globale K-Satz ist ausreichend."
            )

        # Kalibrierung mit Polynomkoeffizienten speichern
        a_g, b_g = st.session_state.get("ab_global", (1.0, 0.0))
        if st.button("Kalibrierung mit T_a-Polynom speichern (JSON)"):
            model = SkyTemperatureModel(
                K_global=st.session_state["K_global"].tolist(),
                a=float(a_g), b=float(b_g),
                poly_coeffs={k: np.array(v) for k, v in poly_coeffs.items()},
                poly_degree=int(poly_degree),
                ta_range=(float(Ta.min()), float(Ta.max())),
                metadata={
                    "created":    datetime.now().isoformat(timespec="seconds"),
                    "source":     "bin_optimization",
                    "n_samples":  int(len(Ta)),
                    "n_bins":     n_bins,
                    "poly_degree": poly_degree,
                },
            )
            model.to_json(CALIBRATION_JSON)
            st.success(f"Gespeichert: `{CALIBRATION_JSON}`")

# ---------------------------------------------------------------------------
# 3. K-Werte für einen bestimmten T_a-Bereich abrufen
# ---------------------------------------------------------------------------
st.header("3. K-Werte für einen Außentemperaturbereich")
st.markdown(
    "Geben Sie den erwarteten T_a-Bereich (z. B. aktuelle Nachttemperatur) ein. "
    "Das Tool interpoliert K1–K7 aus der Polynomkurve K_i(T_a) (Abschnitt 2) "
    "und zeigt den optimalen Parametersatz für diesen Bereich."
)

if "poly_coeffs" not in st.session_state:
    st.info("Bitte zuerst die **Bin-Analyse** (Abschnitt 2) durchführen.")
else:
    col_ta1, col_ta2 = st.columns(2)
    ta_input_min = col_ta1.number_input(
        "T_a min [°C]", value=float(round(Ta.min(), 1)), step=0.5
    )
    ta_input_max = col_ta2.number_input(
        "T_a max [°C]", value=float(round(Ta.max(), 1)), step=0.5
    )
    ta_center = (ta_input_min + ta_input_max) / 2.0

    poly_coeffs = st.session_state["poly_coeffs"]
    poly_degree = st.session_state["poly_degree"]

    K_interp = np.array([
        np.polyval(poly_coeffs[f"K{i+1}"], ta_center) for i in range(7)
    ])

    st.subheader(f"Interpolierte K-Werte  (T_a Mitte = {ta_center:.1f} °C)")
    result_df = pd.DataFrame({
        "K":       [f"K{i+1}" for i in range(7)],
        "Wert":    np.round(K_interp, 2),
        "Gerundet (int)": np.round(K_interp).astype(int),
    })
    st.table(result_df.set_index("K").T)

    # Qualitätsprüfung: RMSE mit diesen K's auf dem Gesamtdatensatz
    Tsky_interp = compute_Tsky(K_interp, Ta, Ts)
    a_i, b_i, WBG_pred_i, rmse_i = linear_wbg_model(Tsky_interp, WBG)
    rho_i, _ = spearmanr(Tsky_interp, WBG)

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE gesamt [%WBG]", f"{rmse_i:.2f}")
    m2.metric("Spearman ρ",         f"{rho_i:.4f}")
    m3.metric("WBG = a·T_sky + b",  f"a={a_i:.3f},  b={b_i:.2f}")

    # Vergleichsplot
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(Tsky_interp, WBG, c=Ta, cmap="coolwarm",
                    alpha=0.8, edgecolors="none", s=45)
    plt.colorbar(sc, ax=ax, label="T_a [°C]")
    xs = np.linspace(Tsky_interp.min(), Tsky_interp.max(), 200)
    ax.plot(xs, a_i * xs + b_i, "k--", lw=1.5,
            label=f"WBG = {a_i:.2f}·T_sky {b_i:+.2f}")
    ax.set(xlabel="T_sky [°C]", ylabel="WBG [%]",
           title=f"Interpolierte K's für T_a ∈ [{ta_input_min:.1f}, {ta_input_max:.1f}] °C  "
                 f"|  RMSE = {rmse_i:.2f} %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # CSV-Export der interpolierten K's
    csv_k = result_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(
        "Interpolierte K-Werte als CSV",
        data=csv_k, file_name=f"K_Ta_{ta_center:.1f}C.csv", mime="text/csv",
    )
