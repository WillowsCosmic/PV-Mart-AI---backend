# ingest.py
import pandas as pd

df = pd.DataFrame({
    "location_id": ["site1", "site1"],
    "year": [2020, 2021],
    "GHI": [450.2, 460.1],
})

df.to_parquet("data/parquet/", partition_cols=["location_id", "year"])