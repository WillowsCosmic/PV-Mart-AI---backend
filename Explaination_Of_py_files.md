### M0 - ingest.py

What it does
Builds a tiny in-memory DataFrame with three columns: location_id, year, and GHI — two rows, both for "site1", one for 2020 and one for 2021.
Writes it to disk as partitioned Parquet under data/parquet/, using partition_cols=["location_id", "year"].
What "partitioned" means here
Rather than writing one flat file, pandas (via pyarrow) splits the data into a folder hierarchy based on the distinct values in the partition columns, dropping those columns from the actual file contents (they get reconstructed from the folder path when read back). For this example, you'd end up with:

ini
data/parquet/
├── location_id=site1/
│   ├── year=2020/
│   │   └── <some-uuid>.parquet   (contains just: GHI=450.2)
│   └── year=2021/
│       └── <some-uuid>.parquet   (contains just: GHI=460.1)


This is exactly the layout every stage in your pipeline (ingest_nasa_power.py, Stage 3, Stage 4, Stage 5, and the monthly aggregator) expects and depends on — they all glob for location_id={id}/year={year} subfolders and read one year-partition at a time. This snippet is essentially a stripped-down illustration of what ingest_location/fetch_year produce at the very start of that chain, minus the NASA API call, retries, and cleaning.

### M1.1 - site_config.py

the origin point of the whole pipeline you've been walking through. It defines the raw site inputs that every later stage (weather ingestion, solar geometry, loss modeling, physics baseline, monthly aggregation) ultimately depends on.

PILOT_SITES
A list of three hardcoded dictionaries, one per pilot rooftop site: Jaisalmer, Kolkata, and Pune. Each entry mixes fields with two very different levels of reliability:

Real/load-bearing: location_id, latitude, longitude, city, azimuth_deg, discom_id (the local electricity distribution utility), tariff_rs_per_kwh (electricity price). These get used directly by pvlib in Stage 3 for solar geometry, so their accuracy matters a lot.
Placeholder, per the docstring: roof_width_ft, roof_depth_ft, parapet_height_ft, obstacles (empty for all three), site_area_sqft. These are explicitly flagged as stand-ins until real customer data comes from an actual site-survey form. This directly explains why Stage 4's shading model was a no-op stub — there's nothing in obstacles to shade anything with yet, since these are placeholder pilot sites, not real surveyed rooftops.
tilt_deg: None for all three sites means none of them override the default tilt — they all fall through to the formula below.

get_effective_tilt(site)
Implements a common rule-of-thumb for fixed-tilt PV arrays: optimal tilt angle ≈ 0.76 × |latitude|. This is a simplified heuristic (real optimal tilt also depends on seasonal irradiance distribution, whether you're optimizing for annual yield vs. winter performance, etc.), but it's a reasonable default when no explicit tilt is specified. The function lets any site override this by setting tilt_deg directly — none currently do.

For these three: Jaisalmer (lat 26.92°) → tilt ≈ 20.44°; Kolkata (lat 22.57°) → tilt ≈ 17.15°; Pune (lat 18.52°) → tilt ≈ 14.08°.

Fit with the rest of the pipeline
This confirms the full picture: Stage 1 defines sites → an NASA POWER ingestion script (mentioned but not shown) pulls raw weather → Stage 3 computes solar geometry/POA using this tilt/azimuth → Stage 4 applies the loss stack (currently with shading disabled, consistent with empty obstacles here) → Stage 5 computes the physics baseline → the monthly aggregation script rolls it up → presumably a Stage 6 AI-correction layer would learn from actual vs. physical output.

### M1.2 - ingest_nasa_power.py

it pulls 26 years (2001 to last full year) of hourly weather/irradiance data from NASA's POWER API for each pilot site, cleans it up, and writes it to the raw partitioned parquet store (data/parquet/) that Stage 3 reads from.

Config section
PARAMETERS maps NASA POWER's cryptic variable codes to physical meanings: ALLSKY_SFC_SW_DWN/DNI/DIFF → GHI/DNI/DHI irradiance components, plus T2M, RH2M, WS10M, PS, PRECTOTCORR for temperature, humidity, wind, pressure, precipitation.
START_YEAR = 2001, END_YEAR dynamically computed as "last full calendar year" — sensible, since the current year's data would be incomplete.
A comment explains the "one year per request" chunking strategy is deliberate: it stays under NASA's per-request day limits and makes caching granular (matches the year-level partition caching used consistently in every other stage you've shown).
Function by function
already_cached — Checks disk for an existing parquet partition before fetching. Same caching pattern seen throughout the pipeline, with the same general caveat: it's a pure existence check, not a validation of completeness or correctness, so a partially-written or corrupted partition would be wrongly treated as "done."

fetch_year — Makes the actual async HTTP GET to NASA POWER for one location/year, with exponential backoff retries (2s, 4s, 8s, 16s) on transport errors or bad HTTP status codes, up to 4 attempts. Reasonable resilience against a flaky public API. If all retries fail, it raises rather than silently skipping — good, since a silent skip here would produce a gap that later stages wouldn't know about.

parse_to_dataframe — Converts NASA's nested JSON (keyed by YYYYMMDDHH timestamp strings) into flat rows, deriving calendar fields (year, month, day, hour, day_of_year) directly from the parsed timestamp — these are exactly the fields Stage 3's cyclic-time-feature function and Stage 4/5's monthly grouping depend on. Timestamps are explicitly tagged UTC.

flag_and_interpolate — Handles NASA POWER's known quirk of using -999 as a missing-data sentinel, converting those to proper NaN, then applying short-gap linear interpolation (capped at 6 consecutive hours) to smooth over brief sensor dropouts without papering over larger data holes. It also computes and reports an "interpolation rate" — the fraction of all cells that got filled in — as a data-quality signal. This is good practice; a comment even correctly justifies why sorting-then-interpolating is safe at this stage (chronological order hasn't been disturbed yet).

ingest_location — Loops years for one site, skips cached years, fetches + parses + cleans + writes for the rest, printing per-year row counts and interpolation rates.

main — Iterates all pilot sites sequentially (not concurrently across sites, though each year within a site is also sequential — this is a deliberate throttling choice, likely to avoid hammering NASA's public API with concurrent requests, though it does mean total runtime scales linearly with sites × years).



### M2 - solar_geometry.py

it takes the raw hourly weather data (pulled from NASA POWER in an earlier ingestion step) and enriches it with time-cyclic features and solar-geometry/irradiance calculations, preparing it for the loss model and physics baseline

add_cyclic_time_features(df)
Converts hour, month, and day_of_year into sine/cosine pairs. This is a standard ML trick: raw hour-of-day (0–23) has a false discontinuity where 23 and 0 are actually adjacent but numerically far apart. Sin/cos encoding maps each cyclic variable onto a circle, so "23:00" and "00:00" end up close together in feature space — useful if these features feed a model later.

add_solar_position_and_poa(df, lat, lon, tilt_deg, azimuth_deg)

Ensures the timestamp column is timezone-aware (UTC).
Uses pvlib.solarposition.get_solarposition to compute, for every hour: solar zenith angle, azimuth, and elevation — i.e., exactly where the sun is in the sky at that location and time.
Feeds those solar angles plus the raw GHI/DNI/DHI (global/direct/diffuse horizontal irradiance) into pvlib.irradiance.get_total_irradiance, which performs the geometric transposition of irradiance from the horizontal plane onto the actual tilted panel surface (given the panel's tilt and azimuth/orientation). This produces POA_irradiance — plane-of-array irradiance — which is the irradiance the panels physically receive, not just what a flat horizontal sensor would see. This is exactly the POA_irradiance column that Stage 5's physics formula uses.
add_cell_temperature(df)
Uses pvlib's Faiman model to estimate the PV cell's operating temperature from POA irradiance, ambient air temperature, and wind speed. This matters because panel efficiency drops as cells heat up — this temperature would typically feed into the eta_losses calculation in Stage 4 (the loss model that Stage 5 reads in).

process_location(site)
For each site:

Looks up the effective tilt angle (via get_effective_tilt, presumably accounting for site-specific mounting) and azimuth.
Iterates over each year of raw data.
Skip-if-exists caching: if a features file for that location/year is already present, it skips reprocessing — same caching pattern as Stage 5, with the same caveat that changing tilt/azimuth config or the pvlib calculation logic won't trigger a recompute unless you delete old outputs first.
Otherwise: loads the raw parquet, restores the location_id/year columns (lost because reading a single partition drops the partition-key columns), applies all three enrichment functions in sequence, then writes the result out to data/features/, partitioned the same way as input.
Prints a quick sanity check per year: row count, max POA irradiance, and max cell temperature — reasonable order-of-magnitude checks (POA maxing out somewhere around 1000–1200 W/m² is physically sensible; a cell temp far outside, say, 0–90°C would suggest a bug).
validate_against_pvlib_reference()
A standalone sanity test, run once at the top of the script before any real processing: it computes solar position for Jaisalmer at a single fixed timestamp near the summer solstice and prints the result with an expected-value comment (~86–87° elevation), letting you eyeball whether pvlib's output matches physical expectations for that latitude and date. This isn't a real assertion/unit test — it just prints numbers for a human to check.

### M3 - loss_model.py

it sits between Stage 3 (solar geometry/POA features) and Stage 5 (physics baseline), and its job is to compute eta_losses, the combined derating factor that Stage 5's formula multiplies against rated power. The docstring makes the design philosophy explicit: rather than one opaque "system losses = 14%" fudge factor (common in simplified PV tools), every loss mechanism is modeled as its own named term so it can be individually inspected, validated, and swapped out later.

Loss term functions
temperature_loss_factor — Standard linear temperature coefficient model: 1 + γ(T_cell - 25°C). Since gamma (γ) is normally negative for PV panels, this correctly produces a factor below 1 when cells run hotter than the 25°C STC reference, and above 1 when cooler. This is the one genuinely physics-based, per-hour term.

shading_loss_factor — This is a stub, not a real implementation. Regardless of the obstacles argument, it always returns an array of 1.0 (no loss). The parameter exists and is documented in the module docstring ("monthly profile per obstacle"), but the actual per-obstacle monthly shading calculation was never written — this silently contributes zero shading loss everywhere. If any pilot site actually has shading obstacles configured, this stage will understate losses for it.

soiling_loss_factor — This one is implemented in a simplified way: dust/dirt accumulates linearly from 0% to SOILING_MAX_PCT (5%) over a fixed 30-day cycle, then resets to 0% (simulating a cleaning). It uses day_of_year % 30 as a proxy for "days since cleaning," which assumes cleanings happen on a perfectly fixed calendar cadence starting from day 1 of the year, every year, forever. The year_start_day parameter is accepted but never used — it's currently a dead parameter (called with None in apply_full_loss_stack), suggesting the original design intended cleaning cycles to be trackable across year boundaries or configurable per-site, but that logic isn't there yet. Also worth noting: this is a fixed idealized sawtooth, not reflecting real weather-driven soiling (e.g., faster buildup in dry/dusty seasons, rain naturally washing panels).

inverter_loss_factor — Models inverter efficiency as load-dependent: efficiency is poor at very low DC input (near cut-in) and approaches rated_eff as load increases, via an exponential curve rated_eff * (1 - 0.05* exp(-8*load)). This is a reasonable simplified proxy for real inverter efficiency curves (which typically dip at low load and plateau near rated capacity), though the specific constants (0.05, 8) look like hand-picked shape parameters rather than being derived from a real inverter datasheet — worth double-checking against actual inverter spec curves if accuracy matters.

module_degradation_factor — Compounding annual module degradation: (1 - 0.5%)^years_since_install, clocked from INSTALL_YEAR. Standard and reasonable.

battery_round_trip_loss_factor — Returns the battery's round-trip efficiency if present, else 1.0 (no battery, no loss). Simple pass-through.

battery_degradation_factor — Linear capacity fade to 80% at end-of-life (replacement_yr), then a hard drop to 0.0 at/after replacement year. That zero is a notable modeling choice: it means once a battery hits its nominal replacement year, this factor makes it contribute zero useful degradation multiplier rather than assuming immediate replacement with a fresh battery. Since degradation_factor = module degradation × battery degradation, this would zero out the entire combined degradation factor (and therefore eta_losses) for that year — which seems like a bug rather than intent, since it would make total PV output plus battery output crash to zero at replacement rather than modeling "battery gets swapped, capacity resets." This should probably be flagged and reconsidered.

apply_full_loss_stack
Pulls all the above together, computing each named loss column, then multiplies them all into a single eta_losses column — this is the value Stage 5 reads. Two things stand out:

shading_loss_factor([], ...) is called with a hardcoded empty obstacle list, not something pulled from site config — so even if the shading function were implemented, it currently could never do anything, since no site's actual obstacles are ever passed in.
soiling_loss_factor(..., None) — confirms the unused year_start_day parameter noted above.
process_location
Same caching pattern as Stages 3 and 5 (skip if output parquet already exists — same stale-cache caveat applies here too, arguably more so, since this stage has the most placeholder/incomplete logic likely to change). Loads each year's features, applies the loss stack, writes to data/losses/, and prints a per-year summary of eta_losses mean/min/max plus the degradation factor.

print_sample_breakdown
A diagnostic utility: grabs a single representative daylight hour (POA > 100 W/m², picks the middle one) from the most recent year of losses data and prints every individual loss term for that hour — directly implementing the spec's requirement that terms be inspectable individually rather than folded into one number.

### M4 - physics_baseline.py

it reduces detailed physics-simulation output down to monthly weather/energy summaries per site, for however many pilot locations are configured, and includes built-in validation prints to catch data issues (wrong row counts, implausible annual totals) before moving to the next stage.

compute_p_physical(df, panel, num_panels)

Computes total array rated power in kW: panel wattage × count of panels / 1000.

Applies the formula above row-by-row (vectorized) to get P_physical_kW.

Clips negative values to zero (irradiance/model noise shouldn't produce negative generation).

Since the data is hourly, kW for that hour numerically equals kWh, so it just copies the column to P_physical_kWh.
process_location(site)
For a given site:

Looks up its panel model, panel count, and total system size (kWp) from SITE_SYSTEM_CONFIG.
Finds all yearly partitions of loss-adjusted data under data/losses/location_id=.../year=*.

For each year:
Caching check: if physics output for that location/year already exists in data/physics/, it just reads that instead of recomputing.
Otherwise, it loads the loss data, tags it with location/year, runs compute_p_physical, and writes it out to data/physics/, partitioned by location and year (Hive-style, like the input).
Computes and prints three sanity metrics: annual kWh, specific yield (kWh per kWp — a standard way to compare systems regardless of size), and CUF (capacity utilization factor — a percentage showing how much of theoretical max continuous output was achieved).

Returns a dict of {year: annual_kWh}.
Main block
Runs this for every pilot site, then prints a cross-location summary of average annual output over the 2001–2025 span.

### M5 - monthly_aggregation.py

It takes hourly (or sub-monthly) physics-based solar/PV simulation data — already computed per location and per year — and rolls it up into monthly summary statistics, then saves the results.

Key pieces

AGG_COLUMNS dictionary

Defines how each column should be aggregated when grouping by month:

Weather/irradiance variables (GHI, DNI, DHI, temperature_C, humidity_pct, wind_speed_m_s, POA_irradiance) are averaged ("mean").

P_physical_kWh (the physics-model's predicted power output) is summed, since energy accumulates over time rather than averaging.

aggregate_location_to_monthly(location_id)

Finds all yearly data folders for a given location under data/physics/location_id=.../year=* (this is a Hive-style partitioned parquet dataset).
For each year: loads the data, groups by month, applies the aggregation rules above, and tags the result with the year and location.
Concatenates all years into one DataFrame, renames P_physical_kWh → PV_energy_kWh, sorts by year/month, and returns a clean DataFrame with a fixed column order.
process_location(site)
Wraps the above for one site: runs the aggregation, writes it out as {location_id}_monthly.parquet in data/monthly/, and prints a sanity-check message comparing the row count to the expected 26 years × 12 months = 312 rows.

(Minor note: len(PILOT_SITES) and 26 * 12 is a slightly odd way to write "2612 if PILOT_SITES is non-empty else 0" — it relies on Python's short-circuit and behavior, which works but is a bit obscure style-wise.)*

Main block

Ensures the output directory exists.
Loops over every site in PILOT_SITES (imported from site_configs), running process_location and storing each site's monthly DataFrame in a dictionary.
Prints a sample of the first 6 rows for one specific location (IN_JSL_001, likely Jaisalmer, India) as a spot-check.
Prints an annual PV energy check — summing the 12 monthly totals back up to yearly totals for 2001 and 2025, for every site — as another sanity check that the aggregation is behaving as expected across the full 26-year span.
Prints a completion message.

### M6 - model_pool.py

Core workflow:-

                    1. Ingest Data          Loads monthly solar generation data per site
                            │
                            ▼
                    2. Feature Prep         Builds cyclical dates (sin/cos) and historical lags (t-1, t-12, rolling 12m)
                            │
                            ▼
                    3. Chronological Split  Holds out the final 3 years for testing (prevents temporal data leakage)
                            │
                            ▼
                    4. Model Tournament     Trains & predicts with 6 distinct candidate approaches:
                                            • Persistence (Naive baseline: same month last year)
                                            • Linear Regression
                                            • Random Forest
                                            • XGBoost
                                            • LightGBM
                                            • SARIMA (Seasonal time-series model)
                            │
                            ▼
                    5. Experiment Tracking  Calculates error metrics (nRMSE, sMAPE, R², MAE, RMSE) and logs
                                            every run, parameter, and score to a local MLflow server
                            │
                            ▼
                    6. Champion Selection   Picks the winning model per site (lowest nRMSE) and saves the 
                                            entire tournament leaderboard to 'scoreboard_m6.csv'
                                
