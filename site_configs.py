"""
Stage 1 — Location & Site Input Layer
Pilot site_config objects for the 3 pilot locations.

These are placeholder values for roof/obstacle fields (fine for pilot testing —
only lat/lng needs to be real). Real customer sites will fill these from
the actual Site Analyser form instead of this hardcoded file.
"""

PILOT_SITES = [
    {
        "location_id": "IN_JSL_001",
        "latitude": 26.9157,
        "longitude": 70.9083,
        "city": "Jaisalmer",
        "roof_width_ft": 40.0,
        "roof_depth_ft": 30.0,
        "parapet_height_ft": 2.0,
        "obstacles": [],  
        "tilt_deg": None,  
        "azimuth_deg": 180.0,  
        "site_area_sqft": 1200.0,
        "discom_id": "JVVNL",
        "tariff_rs_per_kwh": 7.5,
    },
    {
        "location_id": "IN_KOL_001",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "city": "Kolkata",
        "roof_width_ft": 35.0,
        "roof_depth_ft": 25.0,
        "parapet_height_ft": 2.5,
        "obstacles": [],
        "tilt_deg": None,
        "azimuth_deg": 180.0,
        "site_area_sqft": 875.0,
        "discom_id": "CESC",
        "tariff_rs_per_kwh": 8.0,
    },
    {
        "location_id": "IN_PUN_001",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "city": "Pune",
        "roof_width_ft": 38.0,
        "roof_depth_ft": 28.0,
        "parapet_height_ft": 2.0,
        "obstacles": [],
        "tilt_deg": None,
        "azimuth_deg": 180.0,
        "site_area_sqft": 1064.0,
        "discom_id": "MSEDCL",
        "tariff_rs_per_kwh": 7.0,
    },
]


def get_effective_tilt(site: dict) -> float:
    """Stage 1 default tilt formula: beta = 0.76 * |latitude|, user-overridable."""
    if site["tilt_deg"] is not None:
        return site["tilt_deg"]
    return round(0.76 * abs(site["latitude"]), 2)


if __name__ == "__main__":
    for site in PILOT_SITES:
        tilt = get_effective_tilt(site)
        print(f"{site['location_id']} ({site['city']}): "
              f"lat={site['latitude']}, lon={site['longitude']}, tilt={tilt} deg")