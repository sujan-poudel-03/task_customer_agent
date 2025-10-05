# Customer Query Agent

This project explores `Dataset_product_orders.csv` and ships a lightweight agent that answers customer intelligence questions using structured analytics and token-based retrieval.

## Getting Started
1. Ensure Python 3.10+ is available and install dependencies with `pip install -r requirements.txt`.
2. Run `python customer_query_agent.py` to preprocess the dataset, generate JSON artifacts under `artifacts/`, and print sample Q&A traces.
3. Launch the interactive UI with `streamlit run streamlit_app.py` to explore retrieval hits and generated answers.
4. (Optional) Import `build_agent` from `customer_query_agent` to integrate the `QueryAgent` into other workflows.

## Artifacts
- `customer_query_agent.py`: main preprocessing pipeline and query agent.
- `artifacts/preprocessed_data.json`: reusable customer/product profiles, segmentation cutoffs, temporal summaries, and anomaly details.
- `artifacts/knowledge_base.json`: natural-language facts ready for retrieval-augmented answering.
- `artifacts/sample_responses.json`: example questions with answers and supporting reasoning.
- `report.md`: methodology, evaluation, and extension notes.

Refer to `report.md` for a deeper walkthrough of the approach and findings.
