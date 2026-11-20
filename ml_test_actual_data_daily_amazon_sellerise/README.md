# Machine Learning Test - Actual Data - Daily - Amazon - Sellerise

**For testing agents: this is the dataset to pull.** Actual (not forecast) daily Amazon sales, advertising and
inventory for every ASIN in the P&J Trading product catalog, one row per ASIN per calendar day, ready for
machine learning. Source: Sellerise daily exports (sales + ads, ASIN view) and Sellerise Stock By Day (inventory).

| | |
|---|---|
| Main file | `ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet` |
| Key | `asin` + `date` (unique, complete: no missing days) |
| Coverage | 347 ASINs x 1,046 days = 362,962 rows, 2024-01-01 through 2026-11-11 |
| Columns | 109: 2 keys, 4 catalog attributes, 5 flags, 90 Sellerise sales + ads metrics, 8 inventory |
| Units total | 1,931,285 (ties to the raw Sellerise export and to the forecasting actuals, row by row) |

`manifest.json` carries the exact row counts, data-through date and sha256 of each file. Check it before testing.

## Files

- `ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet` - the dataset. Use this one.
- `ml_test_actual_data_daily_amazon_sellerise_sku_day.parquet` - companion, one row per seller SKU per day
  (726 SKUs, sales + ads only). Several SKUs can share one ASIN (FBA and FBM listings); `asin` is on every row.
- `data_dictionary.csv` - every column: source, the header it has in Sellerise, type, missing-day rule.
- `product_catalog.csv` - ASIN, seller SKU, scent, category, size.
- `sample_one_asin_last_120_days.csv` - 120 rows of one ASIN to eyeball the layout without a parquet reader.
- `manifest.json` - build time, coverage, row counts, checksums.

## The missing-day rule (read this before modelling)

A missing day is a **0, not an error**. Every numeric cell is filled; there are no nulls in any numeric column.
The flags tell a real 0 from a filled 0:

| Flag | False means |
|---|---|
| `sellerise_row_present` | Sellerise exported no row for that ASIN on that day (no activity). All sales + ads metrics are filled with 0. 1,470 ASIN-days. |
| `inventory_observed` | Stock By Day did not list the ASIN that day. Inventory columns are filled with 0. **This is "not reported", not a confirmed stockout.** 1,966 ASIN-days. |
| `sessions_available` | Sessions and page views are not available: before 2024-06-26 Sellerise has none, and on the newest days they are not posted yet. They read 0. |
| `attribution_available` | Before 2025-03-14 Sellerise has ad cost but no ad-attributed units, orders or sales. They read 0. |
| `provisional` (True) | Newest 2 days of the export: sessions not posted, ad attribution still filling in. Leave them out of a test window. |

Ratio columns (ACoS, TACoS, CTR, conversion, margin, ROI, per-unit values) are as Sellerise exports them:
Sellerise writes 0 when the denominator is 0. Recompute ratios from the count and dollar columns when aggregating;
never average them.

## Known limits

- Units are what was sold. On out-of-stock days sales are capped by supply, so low units with `available` = 0
  understate demand.
- The SKU-day file is 204 units (0.01%) below the ASIN-day file: Sellerise lists a few "Amazon.Found" relisted-stock
  SKUs without an ASIN mapping. The ASIN-day file is complete and is the authority.
- Inventory is at ASIN grain only.
- Money columns are US dollars as reported by Sellerise.

## Load

```python
import pandas as pd
df = pd.read_parquet("ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet")
test = df[~df.provisional]
```

Rebuilt by `src/ml_export/build_ml_test_dataset.py` in the forecasting repo; the build fails if any check
(complete grid, unique keys, no nulls, totals tie to source) does not pass.
