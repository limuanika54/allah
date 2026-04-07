"""
ECO-VISION: BANGLADESH
NDVI + NPP + Ecosystem Services + Future Predictions | 1990-2040
"""
!pip install streamlit pandas numpy scikit-learn folium streamlit-folium rasterio matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
import math
import folium
from streamlit_folium import folium_static
import io
import base64
import os
import tempfile

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

warnings.filterwarnings("ignore")

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="ECO-VISION: BANGLADESH",
    page_icon="https://img.icons8.com/color/48/000000/bangladesh.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Calibri&display=swap');
    *, *::before, *::after { font-family: 'Calibri', 'Arial', sans-serif !important; }
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-header {
        background: linear-gradient(135deg, #1a5f1a 0%, #0d3d0d 100%);
        padding: 2rem; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 2rem;
        border: 1px solid #2e7d32;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d3d0d, #1a5f1a);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2e7d32;
    }
    .warning-box {
        background: #1a1a2e; padding: 1rem;
        border-left: 5px solid #ff9800; border-radius: 8px;
        margin: 1rem 0; color: #ffffff;
    }
    .success-box {
        background: #0d2e0d; padding: 1rem;
        border-left: 5px solid #4caf50; border-radius: 8px;
        color: #ffffff;
    }
    .info-box {
        background: #0d1f2e; padding: 1rem;
        border-left: 5px solid #2196f3; border-radius: 8px;
        color: #ffffff;
    }
    .stSelectbox label, .stSlider label, .stRadio label,
    .stNumberInput label, .stFileUploader label { color: #ffffff !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown p { color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #0a0a0a; }
    .stDataFrame { background-color: #1a1a1a; color: #ffffff; }
    .dataframe { color: #ffffff !important; }
    .dataframe th { color: #ffffff !important; background-color: #1a1a1a !important; }
    .dataframe td { color: #cccccc !important; background-color: #111111 !important; }
    .stDownloadButton button { background-color: #2e7d32; color: white; border: none; }
    .stDownloadButton button:hover { background-color: #1b5e20; }
    div.block-container { padding-top: 2rem; }
    .ndvi-legend {
        position: fixed; bottom: 80px; left: 50px; z-index: 1000;
        background: rgba(0,0,0,0.85); padding: 10px 14px;
        border-radius: 6px; color: white; font-size: 12px;
        border: 1px solid #333;
    }
    .ndvi-legend-bar {
        width: 180px; height: 14px; border-radius: 3px;
        background: linear-gradient(to right,
            #1a3a5c, #5c4a2a, #8b7355, #c4a94d, #8cc63f, #4caf50, #2e7d32, #1b5e20);
    }
    .ndvi-legend-labels {
        display: flex; justify-content: space-between;
        width: 180px; margin-top: 2px; font-size: 10px; color: #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── District Data ───────────────────────────────────────────────────
DISTRICT_AREAS = {
    "Dhaka": 11875, "Faridpur": 2073, "Gazipur": 1800, "Gopalganj": 1490,
    "Kishoreganj": 2689, "Madaripur": 1145, "Manikganj": 1379, "Munshiganj": 955,
    "Narayanganj": 759, "Narsingdi": 1141, "Rajbari": 1119, "Shariatpur": 1181,
    "Tangail": 3414, "Bandarban": 4479, "Brahmanbaria": 1927, "Chandpur": 1704,
    "Chattogram": 5283, "Cumilla": 3085, "Cox's Bazar": 2492, "Feni": 928,
    "Khagrachhari": 2700, "Lakshmipur": 1456, "Noakhali": 3601, "Rangamati": 6116,
    "Bagerhat": 3959, "Chuadanga": 1174, "Jashore": 2606, "Jhenaidah": 1964,
    "Khulna": 4394, "Kushtia": 1601, "Magura": 1049, "Meherpur": 751,
    "Narail": 990, "Satkhira": 3858, "Bogura": 2920, "Chapai Nawabganj": 1703,
    "Joypurhat": 965, "Naogaon": 3436, "Natore": 1896, "Pabna": 2371,
    "Rajshahi": 2407, "Sirajganj": 2498, "Dinajpur": 3438, "Gaibandha": 2179,
    "Kurigram": 2245, "Lalmonirhat": 1247, "Nilphamari": 1547, "Panchagarh": 1405,
    "Rangpur": 2308, "Thakurgaon": 1781, "Jamalpur": 2032, "Mymensingh": 4363,
    "Netrokona": 2744, "Sherpur": 1364, "Habiganj": 2637, "Moulvibazar": 2599,
    "Sunamganj": 3670, "Sylhet": 3490, "Barguna": 1831, "Barishal": 2785,
    "Bhola": 3737, "Jhalokati": 749, "Patuakhali": 3221, "Pirojpur": 1308,
}

DISTRICT_COORDINATES = {
    "Dhaka": (23.8103, 90.4125), "Chattogram": (22.3569, 91.7832),
    "Khulna": (22.8456, 89.5403), "Rajshahi": (24.3745, 88.6042),
    "Rangpur": (25.7439, 89.2752), "Mymensingh": (24.7471, 90.4203),
    "Sylhet": (24.8993, 91.8712), "Barishal": (22.7010, 90.3535),
    "Cumilla": (23.4683, 91.1747), "Tangail": (24.2511, 89.9163),
    "Bogura": (24.8462, 89.3693), "Dinajpur": (25.6272, 88.6331),
    "Jashore": (23.1634, 89.2182), "Narayanganj": (23.6238, 90.4999),
    "Gazipur": (24.0023, 90.4264), "Pabna": (24.0066, 89.2368),
    "Noakhali": (22.8696, 91.0994), "Cox's Bazar": (21.4272, 92.0057),
    "Bandarban": (22.1954, 92.2183), "Rangamati": (22.6333, 92.2000),
    "Sunamganj": (25.0655, 91.3950), "Habiganj": (24.3805, 91.4150),
    "Moulvibazar": (24.4829, 91.7701), "Kishoreganj": (24.4346, 90.7640),
    "Faridpur": (23.6048, 89.8421), "Gopalganj": (23.0101, 89.8215),
    "Madaripur": (23.1717, 90.2025), "Shariatpur": (23.2423, 90.3480),
    "Rajbari": (23.7548, 89.6384), "Manikganj": (23.8556, 90.0001),
    "Munshiganj": (23.5406, 90.5250), "Narsingdi": (23.9242, 90.7275),
    "Brahmanbaria": (23.9575, 91.1089), "Chandpur": (23.2322, 90.6716),
    "Feni": (23.0183, 91.4019), "Lakshmipur": (22.9420, 90.8322),
    "Khagrachhari": (23.1193, 91.9845), "Bagerhat": (22.6556, 89.7856),
    "Chuadanga": (23.6408, 88.8556), "Jhenaidah": (23.5440, 89.1685),
    "Kushtia": (23.9035, 89.1242), "Magura": (23.4855, 89.4170),
    "Meherpur": (23.7551, 88.6331), "Narail": (23.1675, 89.6315),
    "Satkhira": (22.7046, 89.0723), "Chapai Nawabganj": (24.5931, 88.2678),
    "Naogaon": (24.8087, 88.9461), "Natore": (24.4122, 88.9353),
    "Sirajganj": (24.4534, 89.7008), "Joypurhat": (25.0961, 89.0233),
    "Gaibandha": (25.3262, 89.5401), "Kurigram": (25.8077, 89.6390),
    "Lalmonirhat": (25.9026, 89.4448), "Nilphamari": (25.9137, 88.8509),
    "Panchagarh": (26.3333, 88.5500), "Thakurgaon": (26.0333, 88.4667),
    "Jamalpur": (24.9198, 89.9476), "Netrokona": (24.8806, 90.7352),
    "Sherpur": (25.0189, 90.0165), "Barguna": (22.1487, 90.1192),
    "Bhola": (22.6851, 90.6517), "Jhalokati": (22.6406, 90.2008),
    "Patuakhali": (22.3532, 90.3035), "Pirojpur": (22.5813, 90.0015),
}

DIVISIONS = {
    "Dhaka Division": [
        "Dhaka", "Faridpur", "Gazipur", "Gopalganj", "Kishoreganj",
        "Madaripur", "Manikganj", "Munshiganj", "Narayanganj",
        "Narsingdi", "Rajbari", "Shariatpur", "Tangail",
    ],
    "Chattogram Division": [
        "Bandarban", "Brahmanbaria", "Chandpur", "Chattogram", "Cumilla",
        "Cox's Bazar", "Feni", "Khagrachhari", "Lakshmipur",
        "Noakhali", "Rangamati",
    ],
    "Khulna Division": [
        "Bagerhat", "Chuadanga", "Jashore", "Jhenaidah", "Khulna",
        "Kushtia", "Magura", "Meherpur", "Narail", "Satkhira",
    ],
    "Rajshahi Division": [
        "Bogura", "Chapai Nawabganj", "Joypurhat", "Naogaon",
        "Natore", "Pabna", "Rajshahi", "Sirajganj",
    ],
    "Rangpur Division": [
        "Dinajpur", "Gaibandha", "Kurigram", "Lalmonirhat",
        "Nilphamari", "Panchagarh", "Rangpur", "Thakurgaon",
    ],
    "Mymensingh Division": ["Jamalpur", "Mymensingh", "Netrokona", "Sherpur"],
    "Sylhet Division": ["Habiganj", "Moulvibazar", "Sunamganj", "Sylhet"],
    "Barishal Division": [
        "Barguna", "Barishal", "Bhola", "Jhalokati", "Patuakhali", "Pirojpur",
    ],
}

ECO_COEFFICIENTS = {
    "wetland": 14785, "urban": 0, "cropland": 92,
    "vegetation": 232, "dense_vegetation": 969,
}

# ════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def classify_land_cover(ndvi, mean_ndvi, std_ndvi):
    if ndvi < mean_ndvi - 2 * std_ndvi:
        return "wetland", ECO_COEFFICIENTS["wetland"]
    elif ndvi < mean_ndvi - std_ndvi:
        return "urban", ECO_COEFFICIENTS["urban"]
    elif ndvi < mean_ndvi:
        return "cropland", ECO_COEFFICIENTS["cropland"]
    elif ndvi < mean_ndvi + std_ndvi:
        return "vegetation", ECO_COEFFICIENTS["vegetation"]
    else:
        return "dense_vegetation", ECO_COEFFICIENTS["dense_vegetation"]

def calculate_par(doy=182, latitude=23.5, elevation=10, ea=1.5, kt=1.0):
    Gsc = 1367
    declination = 23.45 * math.sin(math.radians(360 / 365 * (doy - 81)))
    phi = math.radians(latitude)
    delta = math.radians(declination)
    cos_theta = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta)
    theta = math.acos(max(-1, min(1, cos_theta)))
    d2 = 1 / (1 + 0.033 * math.cos(2 * math.pi * doy / 365))
    P = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
    W = 0.14 * ea * P + 2.1
    cos_t = max(math.cos(theta), 0.01)
    tau_sw = 0.35 + 0.627 * math.exp(
        -0.00146 * P / (kt * cos_t) - 0.075 * (W / cos_t) ** 0.4
    )
    Ks = (Gsc * cos_theta * tau_sw) / d2
    IPAR = 0.45 * Ks * 0.000001
    PAR = IPAR * 365 * 3600 * (4 / 52.3019031)
    return PAR

def calculate_npp(ndvi, par):
    FPAR = float(np.clip(-0.168 + 1.24 * ndvi, 0.1, 0.95))
    LUE = 1.8
    par_mj = par * 0.225
    npp = LUE * FPAR * par_mj
    return npp, FPAR

def load_exchange_rates():
    rates = {
        1990: 34.57, 1991: 36.60, 1992: 38.95, 1993: 39.57,
        1994: 40.21, 1995: 40.28, 1996: 41.79, 1997: 43.89,
        1998: 46.91, 1999: 49.09, 2000: 52.14, 2001: 55.81,
        2002: 57.89, 2003: 58.15, 2004: 59.51, 2005: 64.33,
        2006: 68.93, 2007: 68.87, 2008: 68.60, 2009: 69.04,
        2010: 69.65, 2011: 74.15, 2012: 81.86, 2013: 78.10,
        2014: 77.64, 2015: 77.95, 2016: 78.47, 2017: 80.44,
        2018: 83.47, 2019: 84.45, 2020: 84.87, 2021: 85.08,
        2022: 91.75, 2023: 106.31, 2024: 115.60,
    }
    base = 115.60
    for y in range(2025, 2041):
        base *= 1.0024
        rates[y] = base
    return rates

def load_carbon_prices():
    prices = {y: 20.0 for y in range(1990, 2025)}
    for y in range(2025, 2041):
        prices[y] = 20.0 * (1.05) ** (y - 2024)
    return prices

def carbon_sequestration(npp, area_ha, year, ex_rates, c_prices):
    carbon_t_ha = (npp / 1000) * 3.67
    total_t = carbon_t_ha * area_ha
    rate = ex_rates.get(year, 120.0)
    price = c_prices.get(year, 20.0)
    rev_bdt = total_t * price * rate
    return {"carbon_tons": total_t, "revenue_cr": rev_bdt / 1e7}

def ndvi_anomaly(current, history):
    mu = float(np.mean(history))
    sigma = float(np.std(history))
    anom = current - mu
    z = anom / sigma if sigma > 0 else 0.0
    if z > 2:
        label = ">2 SD -- Exceptionally high greenness"
        msg = "Exceptionally high greenness (wet conditions, high moisture)"
    elif z > 1:
        label = "1 to 2 SD -- Above normal"
        msg = "Above-normal vegetation greenness"
    elif z > -1:
        label = "-1 to 1 SD -- Normal"
        msg = "Normal vegetation condition"
    elif z > -2:
        label = "-2 to -1 SD -- Moderate stress"
        msg = "Moderate stress or below-normal vegetation"
    else:
        label = "<-2 SD -- Severe drought"
        msg = "Severe drought or vegetation stress"
    return {"mean": mu, "std": sigma, "anomaly": anom, "z": z, "label": label, "msg": msg}

def stress_index(cur_ndvi, prev_ndvi, cur_rate, prev_rate):
    ex_chg = ((cur_rate - prev_rate) / prev_rate) * 100 if prev_rate else 0
    nd_chg = ((cur_ndvi - prev_ndvi) / prev_ndvi) * 100 if prev_ndvi else 0
    si = ex_chg - nd_chg
    if si > 15:
        level, msg = "Severe Stress", "Dollar price increase severely affecting crop greenness"
    elif si > 8:
        level, msg = "Moderate Stress", "Dollar price increase moderately affecting crop greenness"
    elif si > 2:
        level, msg = "Mild Stress", "Minor impact from dollar price increase"
    else:
        level, msg = "Normal", "No significant stress detected"
    return {"ex_chg": ex_chg, "nd_chg": nd_chg, "si": si, "level": level, "msg": msg}

def predict_ndvi(years, values, targets):
    X = np.array(years, dtype=float).reshape(-1, 1)
    y = np.array(values, dtype=float)
    deg = min(3, len(X) - 1)
    poly = PolynomialFeatures(degree=deg)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    Xt = poly.transform(np.array(targets, dtype=float).reshape(-1, 1))
    preds = np.clip(model.predict(Xt), 0.05, 0.90)
    r2 = r2_score(y, model.predict(Xp))
    return preds, r2

# ════════════════════════════════════════════════════════════════════
#  RASTER PROCESSING
# ════════════════════════════════════════════════════════════════════

def _ndvi_colormap():
    colors = ["#1a3a5c", "#5c4a2a", "#8b7355", "#c4a94d", "#8cc63f", "#4caf50", "#2e7d32", "#1b5e20"]
    return mcolors.LinearSegmentedColormap.from_list("ndvi", colors, N=256)

NDVI_CMAP = _ndvi_colormap()

def _downsample(arr, max_dim=2048):
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    scale = max_dim / max(h, w)
    img = Image.fromarray(arr)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)

def raster_to_rgba(ndvi_arr):
    clipped = np.clip(ndvi_arr, -0.2, 0.9)
    norm = (clipped + 0.2) / 1.1
    rgba = NDVI_CMAP(norm)
    mask = np.isnan(ndvi_arr) | (ndvi_arr < -0.5)
    rgba[mask, 3] = 0
    return (rgba * 255).astype(np.uint8)

def raster_stats(ndvi_arr):
    valid = ndvi_arr[~np.isnan(ndvi_arr) & (ndvi_arr >= -0.2)]
    if valid.size == 0:
        return 0.0, 0.0
    return float(valid.mean()), float(valid.std())

def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.crs is not None and src.crs != CRS.from_epsg(4326):
            t, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            out = np.empty((h, w), dtype=np.float32)
            reproject(source=data, destination=out, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=t, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
            data = out
            left, top = t[2], t[5]
            right = left + w * t[0]
            bottom = top + h * t[4]
            bounds = (bottom, left, top, right)
        else:
            b = src.bounds
            bounds = (b.bottom, b.left, b.top, b.right)
    return data, bounds

def save_temp(uploaded):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name

# ════════════════════════════════════════════════════════════════════
#  MAP & CHART HELPERS
# ════════════════════════════════════════════════════════════════════

def _legend_html():
    return """
    <div class="ndvi-legend">
        <div style="font-weight:bold;margin-bottom:4px;">NDVI</div>
        <div class="ndvi-legend-bar"></div>
        <div class="ndvi-legend-labels">
            <span>-0.2</span><span>0.0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.9</span>
        </div>
    </div>"""

def make_map_with_raster(bounds, rgba_arr, district_name, district_coord, opacity=0.75):
    south, west, north, east = bounds
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10, tiles="OpenStreetMap")
    img = Image.fromarray(rgba_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{b64}", bounds=[[south, west], [north, east]], opacity=opacity, interactive=True).add_to(m)
    if district_coord:
        folium.Marker(district_coord, popup=f"<b>{district_name}</b>", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html()))
    return m

def make_simple_map(district, ndvi_val=None, status=None):
    m = folium.Map(location=[23.685, 90.356], zoom_start=7, tiles="OpenStreetMap")
    coord = DISTRICT_COORDINATES.get(district)
    if coord:
        color = "red" if status and "Severe" in status else ("orange" if status and "Moderate" in status else "green")
        popup = f"<b>{district}</b><br>NDVI: {ndvi_val:.3f}" if ndvi_val else f"<b>{district}</b>"
        folium.Marker(coord, popup=popup, icon=folium.Icon(color=color, icon="leaf")).add_to(m)
    return m

def get_altair_theme():
    return {
        "background": "#000000",
        "view": {"fill": "#000000"},
        "axis": {"domainColor": "#333", "gridColor": "#222", "tickColor": "#888", "labelColor": "white", "titleColor": "white"},
        "legend": {"labelColor": "white", "titleColor": "white"},
        "title": {"color": "white"}
    }

# ════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════
if "rasters" not in st.session_state:
    st.session_state.rasters = {}
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

def _get_all_years():
    if st.session_state.rasters:
        return sorted(st.session_state.rasters.keys())
    if st.session_state.csv_data is not None:
        return sorted(st.session_state.csv_data["year"].tolist())
    return []

def _get_series():
    if st.session_state.rasters:
        yrs = sorted(st.session_state.rasters.keys())
        ndvis = [st.session_state.rasters[y]["mean"] for y in yrs]
        return yrs, ndvis
    if st.session_state.csv_data is not None:
        d = st.session_state.csv_data
        return d["year"].tolist(), d["ndvi"].tolist()
    return [], []

# ════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════
def main():
    ex_rates = load_exchange_rates()
    c_prices = load_carbon_prices()
    par_val = calculate_par()

    st.markdown(
        """<div class="main-header">
            <h1 style="margin:0;">ECO-VISION: BANGLADESH</h1>
            <p style="font-size:1.15rem; margin:6px 0 2px 0;">NDVI + NPP + Ecosystem Services + Future Predictions | 1990-2040</p>
            <p style="font-size:0.95rem; margin:0; color:#b0d0b0;">Satellite-based natural capital valuation for 64 districts</p>
        </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Control Panel")
        division = st.selectbox("Select Division", list(DIVISIONS.keys()))
        district = st.selectbox("Select District", DIVISIONS[division])
        area_km2 = DISTRICT_AREAS.get(district, 1000)
        area_ha = area_km2 * 100
        st.info(f"District Area: {area_km2:,} km\u00b2 ({area_ha:,} ha)")

        st.markdown("---")
        st.markdown("### Data Input Method")
        input_mode = st.radio("Choose method", ["GeoTIFF Raster Upload", "CSV Data Upload"])

        if input_mode == "GeoTIFF Raster Upload":
            if not HAS_RASTERIO:
                st.markdown('<div class="warning-box"><b>rasterio is not installed.</b><br>Install it with: <code>pip install rasterio</code></div>', unsafe_allow_html=True)
            else:
                st.caption("Upload at least 5 GeoTIFF rasters (single-band NDVI) for consecutive years or with a fixed interval.")
                raster_year = st.number_input("Year for this raster", 1990, 2024, 2020, key="ryear")
                raster_file = st.file_uploader("Select GeoTIFF (.tif / .tiff)", type=[".tif", ".tiff"], key="rfile")
                if st.button("Add Raster", key="add_raster_btn") and raster_file:
                    with st.spinner(f"Processing {raster_year} raster..."):
                        try:
                            tmp_path = save_temp(raster_file)
                            data, bounds = read_raster(tmp_path)
                            mu, sig = raster_stats(data)
                            rgba = raster_to_rgba(data)
                            rgba_ds = _downsample(rgba, 2048)
                            st.session_state.rasters[int(raster_year)] = {"path": tmp_path, "mean": mu, "std": sig, "bounds": bounds, "rgba": rgba_ds}
                            st.success(f"Added {raster_year} -- Mean NDVI: {mu:.4f}")
                        except Exception as e:
                            st.error(f"Failed to read raster: {e}")

                if st.session_state.rasters:
                    st.markdown(f"**Uploaded rasters: {len(st.session_state.rasters)}**")
                    yrs = sorted(st.session_state.rasters.keys())
                    for y in yrs:
                        r = st.session_state.rasters[y]
                        col_a, col_b = st.columns([3, 1])
                        col_a.write(f"{y}: NDVI = {r['mean']:.4f} (+/- {r['std']:.4f})")
                        if col_b.button("X", key=f"del_{y}"):
                            try: os.unlink(r["path"])
                            except OSError: pass
                            del st.session_state.rasters[y]
                            st.rerun()

                    if st.button("Clear All Rasters"):
                        for r in st.session_state.rasters.values():
                            try: os.unlink(r["path"])
                            except OSError: pass
                        st.session_state.rasters.clear()
                        st.rerun()

                    if len(yrs) >= 2:
                        diffs = [yrs[i + 1] - yrs[i] for i in range(len(yrs) - 1)]
                        unique_diffs = set(diffs)
                        if len(unique_diffs) == 1:
                            st.info(f"Interval: every {diffs[0]} year(s)")
                        else:
                            st.warning("Years are not evenly spaced.")
        else:
            csv_file = st.file_uploader("Upload CSV (columns: year, ndvi)", type=["csv"], key="csv_up")
            if csv_file:
                try:
                    df_csv = pd.read_csv(csv_file)
                    if "year" not in df_csv.columns or "ndvi" not in df_csv.columns:
                        st.error("CSV must have 'year' and 'ndvi' columns.")
                    elif len(df_csv) < 5:
                        st.error("CSV must contain at least 5 rows.")
                    else:
                        df_csv = df_csv.sort_values("year").reset_index(drop=True)
                        diffs = list(df_csv["year"].diff().dropna())
                        unique = set(diffs)
                        if len(unique) == 1:
                            st.success(f"Loaded {len(df_csv)} records -- Interval: every {int(diffs[0])} year(s)")
                        else:
                            st.warning(f"Loaded {len(df_csv)} records -- Years are not evenly spaced.")
                        st.session_state.csv_data = df_csv
                except Exception as e:
                    st.error(f"CSV read error: {e}")

        st.markdown("---")
        if (st.session_state.rasters and len(st.session_state.rasters) >= 5) or (st.session_state.csv_data is not None and len(st.session_state.csv_data) >= 5):
            all_yrs = _get_all_years()
            if all_yrs:
                sel_year = st.slider("Select Year for Analysis", min(all_yrs), max(all_yrs), max(all_yrs))
                pred_until = st.slider("Predict until year", 2026, 2040, 2035)
        st.markdown("---")
        st.info("Protect vegetation = Protect income")

    years_list, ndvi_list = _get_series()

    if not years_list or len(years_list) < 5:
        st.markdown("""<div class="info-box"><h3>Getting Started</h3><ol><li>Select a division and district from the sidebar.</li><li>Upload at least 5 GeoTIFF rasters (one per year) or a CSV file with 'year' and 'ndvi' columns.</li><li>Rasters must be single-band NDVI GeoTIFFs. Years should be consecutive or follow a fixed interval.</li><li>Once loaded, use the year slider and explore the dashboard.</li></ol><p><b>CSV format example:</b></p><pre style="background:#111;padding:8px;border-radius:4px;">year,ndvi\n2015,0.52\n2016,0.55\n2017,0.50\n2018,0.54\n2019,0.58</pre></div>""", unsafe_allow_html=True)
        return

    df = pd.DataFrame({"year": years_list, "ndvi": ndvi_list}).sort_values("year")
    mean_ndvi = float(np.mean(ndvi_list))
    std_ndvi = float(np.std(ndvi_list))

    if sel_year in df["year"].values:
        cur_ndvi = float(df[df["year"] == sel_year]["ndvi"].values[0])
    else:
        p_tmp, _ = predict_ndvi(df["year"].tolist(), df["ndvi"].tolist(), [sel_year])
        cur_ndvi = float(p_tmp[0])

    future_yrs = list(range(max(df["year"]) + 1, pred_until + 1))
    preds, r2 = predict_ndvi(df["year"].tolist(), df["ndvi"].tolist(), future_yrs)

    anom = ndvi_anomaly(cur_ndvi, ndvi_list)
    lc, eco_c = classify_land_cover(cur_ndvi, mean_ndvi, std_ndvi)
    npp_val, fpar_val = calculate_npp(cur_ndvi, par_val)
    carb = carbon_sequestration(npp_val, area_ha, sel_year, ex_rates, c_prices)
    eco_val = eco_c * area_ha / 1e7
    total_val = carb["revenue_cr"] + eco_val

    st.markdown("## Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Current NDVI</h3><h2>{cur_ndvi:.3f}</h2><p>{anom["label"]}</p><small>Mean: {mean_ndvi:.3f} | Std: {std_ndvi:.3f}</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>Total Economic Value</h3><h2>{total_val:.1f}</h2><p>Crore Taka / year</p><small>Carbon + Ecosystem Services</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>Carbon Sequestration</h3><h2>{carb["carbon_tons"]/1e6:.2f}</h2><p>Million tons CO2e</p><small>${c_prices.get(sel_year,20):.1f} / ton</small></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>Land Cover</h3><h2>{lc.replace("_"," ").title()}</h2><p>Eco Value: {eco_c:,} Tk/ha</p><small>{eco_val:.1f} Cr Tk/year</small></div>', unsafe_allow_html=True)

    if anom["z"] < -1:
        st.markdown(f'<div class="warning-box"><h4>NDVI Anomaly Alert</h4><p>{anom["msg"]}</p><p>Anomaly: {anom["anomaly"]:.3f} (Z-score: {anom["z"]:.2f})</p></div>', unsafe_allow_html=True)

    st.markdown("## NDVI Map Viewer")
    has_raster_data = bool(st.session_state.rasters)

    if has_raster_data:
        view_yr = st.selectbox("Select year to display on map", sorted(st.session_state.rasters.keys()), key="map_yr_select")
        rd = st.session_state.rasters[int(view_yr)]
        coord = DISTRICT_COORDINATES.get(district)
        fmap = make_map_with_raster(rd["bounds"], rd["rgba"], district, coord)
        col_map, col_stats = st.columns([2, 1])
        with col_map:
            folium_static(fmap, width=700, height=500)
        with col_stats:
            st.markdown(f"**Year:** {view_yr}")
            st.markdown(f"**Mean NDVI:** {rd['mean']:.4f}")
            st.markdown(f"**Std NDVI:** {rd['std']:.4f}")
            b = rd["bounds"]
            st.markdown(f"**Bounds:**\n- South: {b[0]:.4f}\n- West: {b[1]:.4f}\n- North: {b[2]:.4f}\n- East: {b[3]:.4f}")
    else:
        fmap = make_simple_map(district, cur_ndvi, anom["label"])
        folium_static(fmap, width=700, height=500)
        st.caption("Upload GeoTIFF rasters to view NDVI maps here.")

    # --- INTERACTIVE CHARTS USING ALTAR (Built-in Streamlit) ---
    st.markdown("## Trend Analysis")
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        hist_df = pd.DataFrame({"year": df["year"], "ndvi": df["ndvi"], "Type": "Historical"})
        pred_df = pd.DataFrame({"year": future_yrs, "ndvi": preds, "Type": "Predicted"})
        chart_df = pd.concat([hist_df, pred_df])
        
        ndvi_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ndvi:Q", title="NDVI", scale=alt.Scale(zero=False)),
            color=alt.Color("Type:N", scale=alt.Scale(range=["#4CAF50", "#FF9800"])),
            strokeDash=alt.StrokeDash("Type:N", scale=alt.Scale(range=[[], [5,5]])),
            tooltip=["year", "ndvi", "Type"]
        ).properties(width=550, height=400, title=f"NDVI Trend -- {district} (R-squared: {r2:.3f})")
        
        st.altair_chart(ndvi_chart.configure(**get_altair_theme()), use_container_width=True)

    with col_ch2:
        revs = []
        for _, row in df.iterrows():
            n, _ = calculate_npp(row["ndvi"], par_val)
            c = carbon_sequestration(n, area_ha, int(row["year"]), ex_rates, c_prices)
            revs.append(c["revenue_cr"])
            
        rev_df = pd.DataFrame({"year": df["year"], "Revenue (Cr Tk)": revs})
        carb_chart = alt.Chart(rev_df).mark_bar(color="#4CAF50").encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Revenue (Cr Tk):Q", title="Revenue (Cr Tk)"),
            tooltip=["year", "Revenue (Cr Tk)"]
        ).properties(width=550, height=400, title="Carbon Credit Revenue (Historical)")
        
        st.altair_chart(carb_chart.configure(**get_altair_theme()), use_container_width=True)

    st.markdown("## Economic Breakdown")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        pie_df = pd.DataFrame({
            "Service": ["Carbon Credits", "Land Cover Services", "Climate Regulation (est.)"],
            "Value": [carb["revenue_cr"], eco_val, eco_val * 0.3]
        })
        pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=80).encode(
            theta=alt.Theta(field="Value", type="quantitative"),
            color=alt.Color(field="Service", type="nominal", scale=alt.Scale(range=["#4CAF50", "#FF9800", "#2196F3"])),
            tooltip=["Service", "Value"]
        ).properties(width=400, height=400, title="Ecosystem Service Value Distribution")
        
        st.altair_chart(pie_chart.configure(**get_altair_theme()), use_container_width=True)

    with col_e2:
        bar_df = pd.DataFrame({
            "Service": ["Carbon Credits", "Land Cover", "Climate Reg."],
            "Value": [carb["revenue_cr"], eco_val, eco_val * 0.3]
        })
        bar_chart = alt.Chart(bar_df).mark_bar(color="#FF9800").encode(
            x=alt.X("Service:N", title="Service"),
            y=alt.Y("Value:Q", title="Value (Cr Tk/year)"),
            tooltip=["Service", "Value"]
        ).properties(width=400, height=400, title="Service Values (Cr Tk/year)")
        
        st.altair_chart(bar_chart.configure(**get_altair_theme()), use_container_width=True)

    if len(df) >= 2:
        prev_row = df[df["year"] < sel_year].tail(1)
        if len(prev_row):
            prev_y = int(prev_row["year"].values[0])
            prev_n = float(prev_row["ndvi"].values[0])
            si = stress_index(cur_ndvi, prev_n, ex_rates.get(sel_year, 100), ex_rates.get(prev_y, 100))
            st.markdown("## Agricultural Stress Index")
            box_cls = "warning-box" if si["level"] != "Normal" else "success-box"
            st.markdown(f'<div class="{box_cls}"><h4>Stress Level: {si["level"]}</h4><p>{si["msg"]}</p><p>Exchange rate change: {si["ex_chg"]:+.2f}% | NDVI change: {si["nd_chg"]:+.2f}% | Index: {si["si"]:.2f}</p></div>', unsafe_allow_html=True)

    st.markdown("## Future Projections")
    rows = []
    for i, (y, p) in enumerate(zip(future_yrs, preds)):
        n_pp, _ = calculate_npp(p, par_val)
        c_pp = carbon_sequestration(n_pp, area_ha, y, ex_rates, c_prices)
        chg = ((p - preds[i - 1]) / preds[i - 1] * 100) if i > 0 else None
        rows.append({"Year": y, "Predicted NDVI": f"{p:.3f}", "Change (%)": f"{chg:+.2f}" if chg else "--", "Carbon Revenue (Cr Tk)": f"{c_pp['revenue_cr']:.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("## Download Data")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Historical NDVI (CSV)", df.to_csv(index=False), f"{district}_historical_ndvi.csv", "text/csv")
    with d2:
        st.download_button("Future Predictions (CSV)", pd.DataFrame(rows).to_csv(index=False), f"{district}_future_predictions.csv", "text/csv")
    with d3:
        full = df.copy()
        full["land_cover"] = [classify_land_cover(n, mean_ndvi, std_ndvi)[0] for n in full["ndvi"]]
        st.download_button("Complete Analysis (CSV)", full.to_csv(index=False), f"{district}_complete_analysis.csv", "text/csv")

    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#666;"><p>ECO-VISION: BANGLADESH | Powered by Landsat Satellite Data</p><p><i>Protect vegetation = Protect income</i></p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
