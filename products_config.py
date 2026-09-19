"""
Product database -- full port from PV-Mart-Pro's src/services/productDatabase.js,
so the forecasting engine recognizes every real product the business sells,
not just the 4 panels / 2 inverters used for initial pilot testing.
"""

PANELS = [
    {"brand": "Tata Power", "model": "TP Mono 500", "watt": 500, "eff": 0.200,
     "tech": "Mono PERC", "sqft": 26, "voc": 49.2, "isc": 13.1, "vmp": 41.5,
     "imp": 12.0, "noct": 45, "gamma": -0.004},
    {"brand": "Tata Power", "model": "TP Mono 550", "watt": 550, "eff": 0.205,
     "tech": "Mono PERC", "sqft": 27, "voc": 50.8, "isc": 13.8, "vmp": 42.8,
     "imp": 12.8, "noct": 45, "gamma": -0.004},
    {"brand": "Adani Solar", "model": "Adani Mono 540", "watt": 540, "eff": 0.210,
     "tech": "Mono PERC", "sqft": 27, "voc": 50.2, "isc": 13.6, "vmp": 42.2,
     "imp": 12.8, "noct": 44, "gamma": -0.0038},
    {"brand": "Adani Solar", "model": "Adani TOPCon 600", "watt": 600, "eff": 0.220,
     "tech": "TOPCon", "sqft": 28, "voc": 53.5, "isc": 14.2, "vmp": 45.0,
     "imp": 13.3, "noct": 43, "gamma": -0.003},
    {"brand": "Adani Solar", "model": "Adani Bifacial 650", "watt": 650, "eff": 0.225,
     "tech": "Bifacial TOPCon", "sqft": 29, "voc": 56.0, "isc": 14.8, "vmp": 47.2,
     "imp": 13.8, "noct": 43, "gamma": -0.003},
    {"brand": "Waaree", "model": "Waaree Mono 540", "watt": 540, "eff": 0.210,
     "tech": "Mono PERC", "sqft": 27, "voc": 50.0, "isc": 13.5, "vmp": 42.0,
     "imp": 12.9, "noct": 45, "gamma": -0.004},
    {"brand": "Waaree", "model": "Waaree Bi-600", "watt": 600, "eff": 0.220,
     "tech": "Bifacial", "sqft": 28, "voc": 53.0, "isc": 14.2, "vmp": 44.8,
     "imp": 13.4, "noct": 43, "gamma": -0.003},
    {"brand": "Vikram Solar", "model": "Somera 540", "watt": 540, "eff": 0.205,
     "tech": "Mono PERC", "sqft": 27, "voc": 49.8, "isc": 13.6, "vmp": 41.8,
     "imp": 12.9, "noct": 45, "gamma": -0.004},
    {"brand": "Vikram Solar", "model": "Somera 550", "watt": 550, "eff": 0.210,
     "tech": "Mono PERC", "sqft": 27, "voc": 50.5, "isc": 13.8, "vmp": 42.5,
     "imp": 12.9, "noct": 45, "gamma": -0.004},
    {"brand": "Loom Solar", "model": "Shark 440", "watt": 440, "eff": 0.200,
     "tech": "Mono PERC", "sqft": 25, "voc": 45.8, "isc": 12.1, "vmp": 38.2,
     "imp": 11.5, "noct": 45, "gamma": -0.004},
    {"brand": "Loom Solar", "model": "Shark 550", "watt": 550, "eff": 0.214,
     "tech": "Mono PERC", "sqft": 26, "voc": 50.5, "isc": 13.8, "vmp": 42.5,
     "imp": 12.9, "noct": 45, "gamma": -0.004},
    {"brand": "LONGi Solar", "model": "Hi-MO 550", "watt": 550, "eff": 0.220,
     "tech": "Mono", "sqft": 27, "voc": 51.0, "isc": 13.6, "vmp": 43.0,
     "imp": 12.8, "noct": 43, "gamma": -0.003},
    {"brand": "LONGi Solar", "model": "Hi-MO 600", "watt": 600, "eff": 0.225,
     "tech": "Mono", "sqft": 28, "voc": 54.0, "isc": 14.0, "vmp": 45.5,
     "imp": 13.2, "noct": 43, "gamma": -0.003},
    {"brand": "Trina Solar", "model": "Vertex 550", "watt": 550, "eff": 0.220,
     "tech": "TOPCon", "sqft": 27, "voc": 51.5, "isc": 13.5, "vmp": 43.5,
     "imp": 12.6, "noct": 43, "gamma": -0.003},
    {"brand": "Trina Solar", "model": "Vertex 670", "watt": 670, "eff": 0.230,
     "tech": "TOPCon", "sqft": 30, "voc": 58.0, "isc": 14.6, "vmp": 49.0,
     "imp": 13.7, "noct": 43, "gamma": -0.003},
    {"brand": "Jinko Solar", "model": "Tiger Neo 620", "watt": 620, "eff": 0.225,
     "tech": "TOPCon", "sqft": 29, "voc": 54.8, "isc": 14.4, "vmp": 46.2,
     "imp": 13.4, "noct": 43, "gamma": -0.003},
    {"brand": "JA Solar", "model": "DeepBlue 550", "watt": 550, "eff": 0.215,
     "tech": "Mono PERC", "sqft": 27, "voc": 50.8, "isc": 13.7, "vmp": 42.8,
     "imp": 12.8, "noct": 45, "gamma": -0.004},
]

INVERTERS = [
    {"brand": "Deye", "model": "SUN-3K-SG03LP1", "kw": 3, "eff": 0.971,
     "mppt": "Dual MPPT", "max_vdc": 500, "min_vdc": 60, "max_vac": 240,
     "type": "hybrid", "dc_ac_ratio": {"min": 1.0, "max": 1.5}},
    {"brand": "Deye", "model": "SUN-5K-SG03LP1", "kw": 5, "eff": 0.978,
     "mppt": "Dual MPPT", "max_vdc": 600, "min_vdc": 90, "max_vac": 240,
     "type": "hybrid", "dc_ac_ratio": {"min": 1.0, "max": 1.5}},
    {"brand": "Deye", "model": "SUN-10K-SG04LP3", "kw": 10, "eff": 0.980,
     "mppt": "Dual MPPT", "max_vdc": 800, "min_vdc": 90, "max_vac": 240,
     "type": "hybrid", "dc_ac_ratio": {"min": 1.0, "max": 1.5}},
    {"brand": "Servotech", "model": "SolarPCU 3K", "kw": 3, "eff": 0.968,
     "mppt": "MPPT", "max_vdc": 480, "min_vdc": 48, "max_vac": 230,
     "type": "hybrid", "dc_ac_ratio": {"min": 0.8, "max": 1.3}},
    {"brand": "Servotech", "model": "SolarPCU 5K", "kw": 5, "eff": 0.972,
     "mppt": "MPPT", "max_vdc": 600, "min_vdc": 90, "max_vac": 230,
     "type": "hybrid", "dc_ac_ratio": {"min": 0.8, "max": 1.3}},
    {"brand": "Visol", "model": "Visol 5K-Grid", "kw": 5, "eff": 0.970,
     "mppt": "Dual MPPT", "max_vdc": 600, "min_vdc": 90, "max_vac": 240,
     "type": "string", "dc_ac_ratio": {"min": 0.9, "max": 1.4}},
    {"brand": "Smarten", "model": "Superb 5K", "kw": 5, "eff": 0.962,
     "mppt": "MPPT", "max_vdc": 580, "min_vdc": 72, "max_vac": 230,
     "type": "string", "dc_ac_ratio": {"min": 0.9, "max": 1.3}},
    {"brand": "Epro", "model": "Epro 5K-Hybrid", "kw": 5, "eff": 0.970,
     "mppt": "MPPT", "max_vdc": 600, "min_vdc": 90, "max_vac": 230,
     "type": "hybrid", "dc_ac_ratio": {"min": 0.9, "max": 1.4}},
    {"brand": "UTL Solar", "model": "UTL Gamma 3K", "kw": 3, "eff": 0.960,
     "mppt": "MPPT", "max_vdc": 450, "min_vdc": 48, "max_vac": 230,
     "type": "string", "dc_ac_ratio": {"min": 0.8, "max": 1.3}},
    {"brand": "UTL Solar", "model": "UTL Gamma 5K", "kw": 5, "eff": 0.972,
     "mppt": "Dual MPPT", "max_vdc": 600, "min_vdc": 90, "max_vac": 230,
     "type": "hybrid", "dc_ac_ratio": {"min": 0.9, "max": 1.4}},
]

BATTERIES = [
    {"brand": "Luminous", "model": "Lithium 5kWh", "kwh": 5.0, "type": "Lithium",
     "eff": 0.95, "v": 48, "dod": 0.90, "cycles": 4000, "replacement_yr": 10},
    {"brand": "Luminous", "model": "Lithium 10kWh", "kwh": 10.0, "type": "Lithium",
     "eff": 0.95, "v": 48, "dod": 0.90, "cycles": 4000, "replacement_yr": 10},
    {"brand": "Exide", "model": "Tubular 150Ah", "kwh": 1.8, "type": "Lead Acid",
     "eff": 0.85, "v": 12, "dod": 0.50, "cycles": 1200, "replacement_yr": 5},
    {"brand": "Exide", "model": "Tubular 200Ah", "kwh": 2.4, "type": "Lead Acid",
     "eff": 0.85, "v": 12, "dod": 0.50, "cycles": 1200, "replacement_yr": 5},
    {"brand": "Okaya", "model": "Lithium 5kWh", "kwh": 5.0, "type": "Lithium",
     "eff": 0.95, "v": 48, "dod": 0.90, "cycles": 3500, "replacement_yr": 10},
    {"brand": "Okaya", "model": "Lithium 7.5kWh", "kwh": 7.5, "type": "Lithium",
     "eff": 0.95, "v": 48, "dod": 0.90, "cycles": 3500, "replacement_yr": 10},
    {"brand": "UTL Solar", "model": "Lithium Pack 5kWh", "kwh": 5.0, "type": "Lithium",
     "eff": 0.95, "v": 48, "dod": 0.90, "cycles": 3500, "replacement_yr": 10},
]


def find_by_model(products: list[dict], model: str) -> dict:
    match = next((p for p in products if p["model"] == model), None)
    if match is None:
        raise ValueError(f"No product found with model '{model}'")
    return match


DEFAULT_PANEL = find_by_model(PANELS, "Adani Mono 540")
DEFAULT_INVERTER = find_by_model(INVERTERS, "UTL Gamma 5K")
DEFAULT_BATTERY = None


def compute_system_size(panel: dict, inverter: dict, dc_ac_ratio: float = 1.15) -> dict:
    target_dc_kw = inverter["kw"] * dc_ac_ratio
    num_panels = round((target_dc_kw * 1000) / panel["watt"])
    return {"num_panels": num_panels, "system_kwp": (num_panels * panel["watt"]) / 1000}


SITE_SYSTEM_CONFIG = {
    "IN_JSL_001": {"panel": DEFAULT_PANEL, "inverter": DEFAULT_INVERTER, "battery": DEFAULT_BATTERY},
    "IN_KOL_001": {"panel": DEFAULT_PANEL, "inverter": DEFAULT_INVERTER, "battery": DEFAULT_BATTERY},
    "IN_PUN_001": {"panel": DEFAULT_PANEL, "inverter": DEFAULT_INVERTER, "battery": DEFAULT_BATTERY},
}

for _location_id, _config in SITE_SYSTEM_CONFIG.items():
    _config.update(compute_system_size(_config["panel"], _config["inverter"]))