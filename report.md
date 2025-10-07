# LLM-Assisted Customer Intelligence System

## 1. Why This Project Exists
`Dataset_product_orders.csv` serves as the company’s collective memory: **1,000 order lines**, **50 customers**, and **100 products** purchased between **18 Feb 2025 and 19 Aug 2025**. The engagement aimed to give teams a helpful companion that can answer natural-language questions about this data without spinning up heavy infrastructure. Along the way the project team:
- cleaned and organised the dataset so customer and product stories are easy to read
- built lightweight embeddings and structured facts for retrieval
- wrapped everything in an agent that speaks plain English while staying grounded in the numbers

## 2. How the Data Was Prepared
### 2.1 Making the rows trustworthy
Python’s built-in `csv` module ingests the file and converts numeric columns to the right types. Dates come through as real `datetime.date` objects, and the pipeline derives a `Month` field (first day of every month) to speed up time-based summaries.

### 2.2 Bringing customers and products to life
- **Customer profiles** stitch together total spend, orders, items, average ticket size, and favourite categories. Unique orders are approximated by pairing each `(customer_id, date, pricelist)` trio.
- **Product profiles** summarise quantity, revenue, typical price, and taxonomy labels. These snapshots later power recommendations and narrative answers.

### 2.3 Segmenting customers for perspective
Quartiles on total spend reveal tiers around **$5,101** (mid) and **$6,674** (high). These cutoffs describe shopper behaviour and limit certain analyses—like “bought together” facts—to the highest-value group when it makes sense.

### 2.4 Tracking time and hierarchy
Monthly revenue is tallied so trends are obvious (for example, `2025-03` ≈ **$48.6K**, `2025-04` ≈ **$45.4K**, `2025-07` ≈ **$48.8K**). The taxonomy becomes a nested dictionary that mirrors *Broad Category → Product Category → Product Sub-Category*. That way the agent can climb up or down the ladder without recomputing anything.

### 2.5 Finding products that travel together
Without explicit order IDs, the implementation treats each `(Customer_ID, Date, Pricelist)` as a purchase session. Unique product combos feed a co-occurrence counter, which later answers “what else did people buy with this?” in both global and high-value contexts.

### 2.6 Spotting behaviour shifts
The workflow compares the latest three months (Jun–Aug 2025) against the earlier period. When a category jumps more than 2× or goes from zero to meaningful share, it raises a flag. That lens surfaced stories like Bonnie Garrett suddenly investing 14% of spend in Food or Brian Turner tripling his Beverage share.

## 3. Storing Knowledge for the Agent
- `artifacts/knowledge_base.json` holds narrative-friendly records about customers, product bundles, and anomalies.
- Each fact is tokenised, lowercased, and stored as a normalised bag-of-words vector. Incoming questions get the same treatment, and cosine similarity picks the best match.
- `artifacts/preprocessed_data.json` keeps all structured profiles, segment thresholds, category hierarchies, monthly revenue, and anomaly markers so other tools can reuse the prep work.

## 4. How the Agent Answers Questions
### 4.1 Orchestrating responses
`QueryAgent` (defined in `customer_query_agent.py`) layers simple pattern recognisers with retrieval. It first checks for phrases that hint at known intents—frequent purchases, bundles, or anomalies—and runs the matching handler. If nothing fits, it falls back to the similarity search over the knowledge base.

### 4.2 Matching people and categories gracefully
Names and IDs are fuzzy-matched so typos don’t derail the experience. Category lookups lean on the prebuilt hierarchy, letting the agent explain answers at whatever level the user referenced.

### 4.3 Using the Gemini API when available
When the environment variable `GEMINI_API_KEY` is set, the agent calls the modern `google-genai` client (`gemini-2.5-flash` by default) to polish responses or return structured JSON. Without the key, it still shares plain-language narratives assembled from the structured data.

## 5. How Well It Works Today
### 5.1 Manual checks
Test queries about customer spend, bundle facts, and anomalies pulled back the right entries from the knowledge base. Because the vocabulary is small and factual, cosine similarity behaves predictably.

### 5.2 Benchmark snapshot
The evaluation covers six labelled questions spanning customer profiles, basket bundles, and anomaly detection (`artifacts/benchmark_results.json`). Precision@1 remains **1.00** (6/6 hits), showing that each intent still maps cleanly to the expected knowledge record.

### 5.3 Honest limitations
- Token-based intent detection is fast yet literal; nuanced phrasing may slip through.
- The anomaly detector uses simple share ratios and ignores variance. More statistical treatment (rolling z-scores or Bayesian changepoints) would make flags sturdier.
- Co-occurrence counts are thin because many product pairs appear once. Association rules or collaborative filtering would create richer recommendations.

## 6. Strengths, Safeguards, and Open Questions
- Keeps dependencies light for easy deployment, yet produces JSON artifacts anyone can audit.
- Every answer quotes the underlying metrics so reviewers can trace the logic.
- Future upgrades could add semantic embeddings, multi-fact aggregation, or external vector stores without rewriting the foundation.

## 7. Where to Take It Next
1. Introduce sentence-level embeddings or a lightweight intent classifier for better paraphrase handling.
2. Extend the co-occurrence graph with lift/confidence and explore full association-rule mining.
3. Turn the monthly revenue and anomaly feeds into dashboards for business teams.
4. Generate customer journey briefs by weaving together segmentation, anomalies, and frequent purchases.

## 8. Bonus Challenge Outcomes
The bonus deliverable ships as `bonus_challenge.py`, a self-contained helper that reuses the analytics layer. Two capabilities stood out:
- **Context-aware recommendations**: `BonusAdvisor.recommend_products()` extracts customer IDs, categories, and sub-categories from natural-language prompts. When the scope is specific it filters rows accordingly; otherwise it falls back to global best sellers. Outputs include the rationale context and a machine-consumable recommendation list.
- **Cross-customer reasoning**: `BonusAdvisor.multi_customer_reasoning()` compares spend across multiple customers or categories, filling gaps with top-spending customers when the prompt is sparse. It returns a narrative headline plus structured breakdowns for downstream dashboards.

The script exposes a CLI (`python bonus_challenge.py --mode recommend|compare`) that accepts inline questions or prompts interactively if none are supplied, keeping analysis quick for ad-hoc requests.

## 9. Running the Project
- **Process the data and print sample conversations**: `python customer_query_agent.py`
- **Reuse the agent in other code**: `from customer_query_agent import build_agent`
- **Explore the UI**: `streamlit run streamlit_app.py`
- **Inspect generated artifacts**: open `artifacts/preprocessed_data.json`, `artifacts/knowledge_base.json`, and `artifacts/sample_responses.json`

Together these pieces deliver a helpful assistant that stays grounded in the dataset while giving teams clear, human-ready answers.
