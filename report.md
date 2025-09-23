# LLM-Assisted Customer Intelligence System

## 1. Objectives and Dataset Context
The project combines lightweight natural language understanding with structured analytics to answer customer-buying questions over `Dataset_product_orders.csv`. The dataset spans **1,000 order line items** between **18 Feb 2025 and 19 Aug 2025**, covering **50 customers** and **100 products** across the hierarchical taxonomy *Broad Category -> Product Category -> Product Sub-Category*. Purchase contexts capture list price (`Retail` vs `Wholesale`), per-line revenue, and order-level totals, enabling temporal, categorical, and value-based reasoning.

Primary goals were:
- Build reusable preprocessing pipelines that expose per-customer, per-product, and temporal signals.
- Create compact embeddings/structured representations to support retrieval-augmented question answering.
- Implement an agent that can resolve natural-language prompts about frequent purchases, co-purchase bundles, and anomalous behaviour.
- Document evaluation, limitations, and opportunities for richer LLM integration.

## 2. Data Preparation and Feature Engineering
### 2.1 Cleansing and Type Normalisation
Records were ingested with Python's `csv` library to avoid third-party dependencies. All numeric columns were converted to `int`/`float`; order dates were parsed into `datetime.date` objects. Each row carries a derived `Month` attribute (first day of month) to accelerate temporal grouping.

### 2.2 Entity Profiles
Two profile layers drive downstream reasoning:
- **Customer profiles** capture spend, item volume, ticket size, first/last purchase, category frequency, category spend, and product quantity distributions. The average ticket metric (mean order subtotal) is computed by counting unique `(customer_id, date, pricelist)` tuples as proxy order IDs.
- **Product profiles** consolidate overall quantity, revenue, and average unit price alongside taxonomy labels. These profiles underpin recommendation explanations and category hierarchies.

### 2.3 Segmentation and Thresholds
Customer spend distributions yielded quartile-based cutoffs: `mid approx. $5,101` and `high approx. $6,674`. These thresholds define segments for high-value basket mining and allow the agent to adapt answers (e.g., restrict co-occurrence analysis to high-tier shoppers).

### 2.4 Temporal and Hierarchical Signals
Monthly revenue totals are available for seasonality checks, e.g., `{2025-03: $48.6K, 2025-04: $45.4K, 2025-07: $48.8K}`. Category hierarchies were materialised as nested dictionaries so the agent can reason across `Broad_Category -> Product_Category -> Product_Sub_Category` levels without additional lookups.

### 2.5 Co-occurrence Graph
Because explicit order IDs are absent, orders are approximated by the triplet `(Customer_ID, Date, Pricelist)`. Unique product combinations per order feed a co-occurrence counter. These counts drive the "bought together" recommendations both globally and within the high-value cohort.

### 2.6 Change Detection for Anomalies
Behaviour shifts are flagged by comparing category spend share in the most recent three months (Jun-Aug 2025) against prior months. Ratios >= 2x or previously zero baseline are tagged as unusual. Example: *Bonnie Garrett* moved from 0% to 14% of spend in Food (ratio -> treated as 999x), while *Brian Turner* tripled Beverage share.

## 3. Embedding and Retrieval Strategy
### 3.1 Knowledge Records
The system serialises structured facts into `artifacts/knowledge_base.json`. Records cover:
- Customer summaries with top categories and spend.
- High-signal product bundle facts from the co-occurrence graph.
- Detected anomalous category shifts.

### 3.2 Token-Based Embeddings
Each record text is tokenised (alphanumeric lowercasing) and transformed into l2-normalised bag-of-words vectors. The same pipeline is applied to user questions. Cosine similarity provides lightweight retrieval without external libraries, allowing the agent to surface the most relevant fact for broad queries outside the dedicated heuristics.

### 3.3 Stored Preprocessed Data
A companion artifact, `artifacts/preprocessed_data.json`, stores customer/product profiles, segmentation bins, category hierarchies, monthly revenue, and anomaly metadata. This enables re-use by notebooks, dashboards, or future LLM-powered services without recomputation.

## 4. Query Agent Architecture
### 4.1 Rule-Orchestrated Workflow
`customer_query_agent.py` exposes `QueryAgent`, which sits atop `OrderAnalytics`. Question handling follows a cascading strategy:
1. **Pattern-specific handlers** trigger when key phrases appear:
   - *"frequently/often"* -> frequent purchase summariser that pinpoints the requested customer and category.
   - *"bought/buy together"* -> co-occurrence recommender scoped to high-value customers.
   - *"unusual/unique/anomaly"* -> change detection reporter using the recent-window metrics.
2. **Retrieval fallback** leverages cosine-similarity scoring over the knowledge base facts for more open-ended questions.

### 4.2 Customer and Category Resolution
Customer names and IDs are matched via token overlap. Category detection scans across broad, core, and subcategory labels, ensuring prompts like "Beverages" or "Bakery" can map to the correct slice. The frequent-purchase answer includes spend context and the top contributing products as evidence.

### 4.3 Advanced Insight Hooks
- **Temporal reasoning** uses the month-level aggregates to cite revenue trends (exposed through artifacts for downstream reporting).
- **Category hierarchy** ensures recommendations respect the multi-level taxonomy, helpful for future expansion (e.g., grouping by Broad Category first when subcategory granularity is sparse).
- **Customer segmentation** influences which baskets feed the recommendation engine and forms the basis for premium-customer summaries.

## 5. Evaluation and Example Interactions
### 5.1 Functional Validation
Running `python customer_query_agent.py` generates artifacts and prints exemplar dialogues (also stored in `artifacts/sample_responses.json`). Key outputs:
- **Frequent purchase** -> "Lindsey Glass frequently buys Beverages items. Top picks: Product 47 (19), Product 93 (10), Product 21 (10). Total spend $8,068 across 21 orders (avg ticket $384)."
- **High-value bundles** -> "High-value customers tend to bundle: Product 77 (Beverage) with Product 94 (Beverage) - 1 joint orders; ..."
- **Anomaly detection** -> "Recent anomalies: Bonnie Garrett increased Food share to 14% (was 0%); Brian Turner increased Beverage share to 24% (was 2%); ..."
- **Open query via retrieval** -> e.g., "What are the top categories for Customer_001?" retrieves the relevant customer profile fact with cosine similarity approx. 0.30.

### 5.2 Accuracy Considerations
- The absence of explicit order IDs introduces a mild risk of merging same-day repeat orders. For the available data this approximation still produced coherent frequency and co-occurrence patterns, but additional order keys would improve fidelity.
- Token-based intent detection is deterministic and fast but lacks the linguistic nuance of a full LLM. Expanding the knowledge base and incorporating semantic embeddings (e.g., from sentence transformers) would increase recall for varied phrasings.
- The anomaly detector uses a simple share-ratio threshold and ignores statistical variance. Incorporating rolling z-scores or Bayesian changepoint methods would better calibrate "unusual" signals.

### 5.3 Retrieval Quality
Manual spot checks confirm that the highest-similarity records generally align with query topics. Because the vocabulary is sourced from factual statements, cosine scores are well-calibrated for the limited domain. However, the agent currently caps the output to single retrieved facts; multi-hop reasoning could aggregate several supporting facts in future iterations.

## 6. System Capabilities and Limitations
### 6.1 Strengths
- Entire pipeline is dependency-light and executable in constrained environments.
- Artifacts expose both raw metrics and natural-language-ready facts, enabling downstream reuse.
- Modular design separates analytics (data prep) from interaction (query agent), simplifying future upgrades.

### 6.2 Limitations
- Rule-based NLP may miss intents that rely on paraphrase or implicit context.
- High-value co-occurrence counts are sparse (many pairs occur once). Bootstrapping with broader baskets or collaborative filtering would improve recommendation strength.
- No live LLM integration is present due to offline constraints; responses are template-driven albeit data-grounded.

### 6.3 Risk Mitigations
- All outputs include compact reasoning snippets and supporting facts to aid auditing.
- JSON artifacts can be inspected or versioned to verify consistency across runs.

## 7. Extension Opportunities
1. **Embed real LLMs** (e.g., local or API-backed) to paraphrase retrieved facts, perform intent classification, and chain-of-thought reasoning.
2. **Vector database integration** for scalable similarity search once embeddings expand beyond bag-of-words.
3. **Temporal dashboards** using the stored monthly revenue and anomaly feeds to visualise trends.
4. **Recommendation uplift** by enriching the co-occurrence graph with lift/confidence metrics or deploying association-rule mining.
5. **Customer journeys** that combine segmentation, anomalies, and frequent purchases into automatically generated account briefings.

## 8. Execution Guide
- **Run preprocessing and demo**: `python customer_query_agent.py`
- **Reuse components**: import `build_agent()` to obtain a ready-to-query `QueryAgent` instance.
- **Inspect results**: review `artifacts/preprocessed_data.json`, `artifacts/knowledge_base.json`, and `artifacts/sample_responses.json` for structured outputs and QA logs.

The deliverables collectively demonstrate how structured analytics, light embeddings, and rule-based orchestration can mimic a constrained LLM workflow, providing explainable answers for typical customer intelligence questions without external dependencies.
