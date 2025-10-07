# Structured Customer Insights Report

This document summarises two analytical queries executed over `Dataset_product_orders.csv` using the Groq-assisted pandas workflow. All computations are reproducible via the scripts in `task/structured_rag.py` and the accompanying manual analysis snippet (`task/analysis_temp.py`).

## 1. Frequently Co‑Purchased Products Among High-Value Customers

- **Pipeline steps**
  1. Loaded 1,000 order lines and normalised column names.
  2. Calculated each customer's cumulative spend (`total_subtotal`) and selected the top 10% (threshold ≈ **$8,354.86**).
  3. Built synthetic order identifiers combining `customer_id` and `date`, then collected unique products per order.
  4. Generated all product pairs (combinations of size 2) within each high-value order and counted pair frequency across orders.
- **Result**
  - The data is very sparse—every detected pair appears only once—which suggests high-value customers rarely buy multiple products together in a single order.
  - Representative pairs:
    | Product A   | Product B   | Orders |
    |-------------|-------------|--------|
    | Product 77  | Product 85  | 1      |
    | Product 77  | Product 94  | 1      |
    | Product 20  | Product 82  | 1      |
    | Product 24  | Product 84  | 1      |
    | Product 24  | Product 96  | 1      |
  - **Takeaway:** High-value accounts tend to place concentrated, single-SKU orders. Promotional strategies aimed at bundles may need additional incentives or cross-selling outreach.

## 2. Unusual / Unique Buying Patterns in the Last 3 Months

- **Pipeline steps**
  1. Determined the recency window as **19 May 2025** onward (three months before the dataset’s max date).
  2. Converted `order_quantity` and `subtotal` to floats and computed z-scores within the filtered window.
  3. Flagged lines where either z-score exceeded 2 (i.e., >2σ from the mean) and ranked by the maximum absolute z-score across both metrics.
- **Result**
  - 15 standout orders exhibited exceptionally high quantities or spend. Examples:
    | Date       | Customer        | Product      | Quantity | Subtotal ($) | Max Z-score |
    |------------|-----------------|--------------|----------|--------------|-------------|
    | 2025-05-30 | William Rios    | Product 100  | 10       | 983.10       | 3.07        |
    | 2025-07-19 | Edward Moreno   | Product 85   | 10       | 957.30       | 2.96        |
    | 2025-06-01 | Leslie Murray   | Product 64   | 10       | 954.10       | 2.94        |
    | 2025-08-16 | Heather Gray    | Product 30   | 10       | 929.70       | 2.84        |
    | 2025-07-30 | Bonnie Garrett  | Product 27   | 10       | 925.10       | 2.82        |
  - **Takeaway:** These spikes come from full-cart replenishments (quantity 9–10) on premium SKUs. They merit investigation for promotion timing, inventory checks, or VIP outreach.

## Execution Notes

- Computational outputs were generated via `task/analysis_temp.py` (run inside the project’s virtual environment with `PYTHONIOENCODING=utf-8`).
- The Groq-assisted query planner in `task/structured_rag.py` required retries due to API rate limits and JSON-to-code translation errors. For this report, we relied on a deterministic pandas script to guarantee reproducibility.
- All intermediate artefacts, including the JSON payload summarising thresholds and outlier records, were logged to the console for audit.

For further analysis or automation, the helper script can be converted into a CLI utility or integrated back into the Streamlit interface once Groq rate limits stabilise.
