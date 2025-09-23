# Task Overview

## Data Understanding & Preprocessing
- Perform exploratory analysis of the dataset to understand customer purchasing patterns, product categories, and trends.
- Preprocess the data to make it suitable for querying and knowledge retrieval by an LLM.
- Generate embeddings or structured representations for products, categories, and customer profiles.

## LLM-Powered Query System
- Build an AI agent capable of answering complex customer-related queries, such as:
    - “Which products does Customer_X frequently purchase in the Beverages category?”
    - “Suggest products that are usually bought together by high-value customers.”
    - “Identify unusual or unique buying patterns in the last 3 months.”
- Use LLMs for natural language understanding and integrate with the dataset to generate accurate responses.
- Implement retrieval-augmented generation (RAG) or vector database search for structured data interaction.

## Advanced Features (Challenge)
- Implement reasoning across multiple dimensions:
    - Temporal patterns (seasonality, recency of purchases)
    - Category hierarchies (Broad_Category → Product_Category → Product_Sub_Category)
    - Customer segmentation insights
- Allow the system to answer questions in natural language using these insights.

## Evaluation
- Evaluate the system for accuracy, relevance, and contextual understanding.
- Provide example queries and responses to demonstrate reasoning ability.
- Optionally, benchmark using embedding similarity scores or precision/recall of retrieved information.

## Deliverables
- A Jupyter Notebook or Python script demonstrating the LLM-powered query system.
- A 2–3-page report summarizing:
    - Data preprocessing and embedding strategies
    - Model architecture and approach for query handling
    - Evaluation results and system capabilities
    - Example interactions (questions and answers) to showcase system performance.

## Optional Bonus Challenge
- Allow the system to generate product recommendations dynamically based on query context.
- Integrate reasoning across multiple customers or categories to answer more complex queries.

---

This task is designed to push the limits of LLM usage in structured data reasoning, testing your ability to combine LLMs, embeddings, and structured datasets for advanced AI applications.
