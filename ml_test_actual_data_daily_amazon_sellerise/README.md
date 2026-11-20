# Machine Learning Test - Actual Data - Daily - Amazon - Sellerise

**For testing agents: this is the dataset to pull.** Actual (not forecast) daily Amazon sales, advertising and
inventory for every ASIN in the P&J Trading product catalog, one row per ASIN per calendar day.

**It is the same data the forecasting system uses.** The main file is the forecasting table `daily_facts`
(the table every forecasting script reads), loaded with the forecasting system's own loader and carried over
column for column, value for value. That table is pieced together from several reports by the forecasting scripts:

| Piece | Report | Script |
|---|---|---|
| Units, sales, ads, sessions per ASIN per day | Sellerise sales + profit daily exports, ASIN view (many exports stitched; the later export wins on shared days) | `src/daily_sales.py` |
| FBA / FBM split of units | Sellerise daily exports, SKU view + SKU-to-ASIN crosswalk (catalog, FBA reports) | `src/daily_sales.py` |
| Inventory and the in-stock / out-of-stock state of each day | Sellerise Stock By Day | `src/sellerise_daily.py`, `src/build_curated.py` |
| Stockout-adjusted units (`units_adj`) | true-demand day flags (expected sales on out-of-stock days) | `src/true_demand.py` |
| Sale-event days, owner-documented shocks | `config/event_calendar.csv`, `config/events.csv` | `src/asin_daily/facts.py` |
| The assembled table | all of the above; monthly totals gated against Amazon Business Reports | `src/asin_daily/facts.py` |

| | |
|---|---|
| Main file | `ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet` |
| Key | `asin` + `date` (unique, complete: no missing days) |
| Coverage | 347 ASINs x 1,046 days = 362,962 rows, 2024-01-01 through 2026-11-11 |
| Columns | 138: 2 keys, 4 catalog attributes, 72 forecasting columns, 2 added flags, 58 added Sellerise columns |
| Units | `units` 1,931,285 sold; `units_adj` 2,009,395 stockout-adjusted (what the models train on) |

`manifest.json` carries the exact row counts, data-through date and sha256 of each file. Check it before testing.

## Which column to model

- `units` - units actually sold. On out-of-stock days it is capped by supply.
- `units_adj` - units with out-of-stock days lifted to expected demand. Equal to `units` on every in-stock day.
  This is the demand series the forecasting system uses.
- `inv_state` - `CLEAN` (in stock), `SOFT_RD` / `SOFT_IP` / `HARD` (out-of-stock states; `is_outage_day` = True),
  `UNOBSERVED` (Stock By Day did not list the ASIN that day).
- `units_fba`, `units_fbm` - the split of `units` by fulfillment.

## Files

- `ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet` - the dataset. Use this one.
- `ml_test_actual_data_daily_amazon_sellerise_sku_day.parquet` - companion, one row per seller SKU per day
  (726 SKUs, Sellerise sales + ads). Its FBA and FBM unit totals equal `units_fba` and `units_fbm` of the main file.
- `data_dictionary.csv` - every column: where it comes from, its Sellerise header, type, blank rule, description.
- `product_catalog.csv` - ASIN, seller SKU, scent, category, size.
- `sample_one_asin_last_120_days.csv` - 120 rows of one ASIN to eyeball the layout without a parquet reader.
- `manifest.json` - build time, coverage, row counts, blank counts, checksums.

## The missing-day rule (read this before modelling)

A missing day is a **0, not an error**. Every ASIN has a row for every day, and every sales and ads count is 0 on a
day Sellerise exported no row (`sellerise_row_present` = False, 1,470 ASIN-days).

Blanks exist only where the forecasting table itself is blank, because a 0 would be a wrong number there:

| Columns | Blank means |
|---|---|
| `available`, `total_stock`, `reserved`, `inbound`, `inbound_shipped`, `inbound_receiving`, `unfulfillable` | Stock By Day did not list the ASIN that day (`inv_state` = `UNOBSERVED`, `inventory_observed` = False, 1,966 ASIN-days). **Not a confirmed stockout.** |
| `asp`, `conversion`, `organic_conversion`, `ppc_conversion_sp`, `cpc_sp`, `efficiency_sp`, `tacos_sp`, `acos_sp` | The denominator is 0 that day; the ratio is not defined. |
| `spell_id`, `day_of_spell`, `expected_fba_B`, `fbm_base` | The day is not part of an out-of-stock spell. |

Flags to respect:

| Flag | Meaning |
|---|---|
| `sessions_available` | False before 2024-06-26: Sellerise has no sessions or page views, they read 0. |
| `attribution_available` | False before 2025-03-14: Sellerise has ad cost but no ad-attributed units, orders or sales, they read 0. |
| `provisional` | True on the newest 2 days: sessions not posted, ad attribution still filling in. Leave them out of a test window. |

The 58 added Sellerise columns (fees, cost of goods, profit detail, per-ad-type detail) are not used by the forecasting
table. Their ratio columns are as Sellerise exports them (0 when the denominator is 0); recompute ratios from the
count and dollar columns when aggregating, never average them. `conversion_sellerise` is the Sellerise-exported
conversion; `conversion` is the forecasting table's units / sessions.

## Known limits

- The SKU-day file is 204 units (0.01%) below the ASIN-day file: Sellerise lists a few "Amazon.Found" relisted-stock
  SKUs without an ASIN mapping. The ASIN-day file is complete and is the authority.
- Inventory is at ASIN grain only.
- Money columns are US dollars as reported by Sellerise.
- Daily history starts 2024-01-01, the first day of the Sellerise daily exports.

## Load

```python
import pandas as pd
df = pd.read_parquet("ml_test_actual_data_daily_amazon_sellerise_asin_day.parquet")
test = df[~df.provisional]
```

Rebuilt by `src/ml_export/build_ml_test_dataset.py` in the forecasting repo. The build fails unless every forecasting
column is identical to `daily_facts` on every row, `daily_facts` reproduces from its sources, the grid is complete,
and the totals tie to the raw Sellerise store.
