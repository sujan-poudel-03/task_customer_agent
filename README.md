# Customer Query Agent

This project explores `Dataset_product_orders.csv` and ships a lightweight agent that answers customer intelligence questions using structured analytics and token-based retrieval.

## Getting Started
1. Ensure Python 3.10+ is installed.
2. Create and activate a local virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   python -m pip install --upgrade pip
   ```
3. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the sample environment file and add your credentials:
   ```bash
   cp .env.example .env
   ```
   Update `GEMINI_API_KEY` with a valid Google Gemini key (or leave it blank to keep rule-based responses only).
5. Run `python customer_query_agent.py` to preprocess the dataset, generate JSON artifacts under `artifacts/`, and print sample Q&A traces.
6. Launch the interactive UI with `streamlit run streamlit_app.py` to explore retrieval hits and generated answers.
7. (Optional) Import `build_agent` from `customer_query_agent` to integrate the `QueryAgent` into other workflows.
8. (Bonus) Explore the advanced utilities with `python bonus_challenge.py`. Add a natural-language question inline or pass `--demo` to run canned examples. Use `--mode compare` to inspect multi-customer reasoning. For example:
   ```bash
   python bonus_challenge.py "Suggest high-margin add-ons for CUST_015 in the Electronics category" --mode recommend --pretty
   python bonus_challenge.py "Compare CUST_008 and CUST_014 across personal care and household essentials" --mode compare --pretty
   python bonus_challenge.py --demo --pretty
   ```

## Artifacts
- `customer_query_agent.py`: main preprocessing pipeline and query agent.
- `artifacts/preprocessed_data.json`: reusable customer/product profiles, segmentation cutoffs, temporal summaries, and anomaly details.
- `artifacts/knowledge_base.json`: natural-language facts ready for retrieval-augmented answering.
- `artifacts/sample_responses.json`: example questions with answers and supporting reasoning.
- `report.md`: methodology, evaluation, and extension notes.

Refer to `report.md` for a deeper walkthrough of the approach and findings.
