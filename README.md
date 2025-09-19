## Pseudo-Industry generation via graph communities

Generate weekly pseudo-industry mapping from HS300-like CSV input.

Example:

```bash
pip install -r requirements.txt
bash scripts/generate_pseudo_industry.sh data/hs300.csv outputs/pseudo_industry
```

This will produce per-week CSVs and `pseudo_industry_latest.parquet/csv` in the output directory.

# FDA