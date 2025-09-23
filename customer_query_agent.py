#!/usr/bin/env python3
"""
Customer query agent for Dataset_product_orders.csv.
Performs preprocessing, builds lightweight embeddings, and answers natural language queries.
"""

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Tuple

DATA_PATH = "Dataset_product_orders.csv"
ARTIFACT_DIR = "artifacts"


def load_orders(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k: v.strip() for k, v in raw_row.items()}
            row["Order_Quantity"] = int(row["Order_Quantity"])
            row["Order_Lines_Unit_Price"] = float(row["Order_Lines_Unit_Price"])
            row["Subtotal"] = float(row["Subtotal"])
            row["Total_OrderQuantity"] = int(row["Total_OrderQuantity"])
            row["Total_Subtotal"] = float(row["Total_Subtotal"])
            row["Date"] = datetime.strptime(row["Date"], "%Y-%m-%d").date()
            row["Month"] = date(row["Date"].year, row["Date"].month, 1)
            rows.append(row)
    return rows


@dataclass
class CustomerProfile:
    customer_id: str
    customer_name: str
    total_spend: float
    total_orders: int
    total_items: int
    avg_ticket: float
    category_counts: Dict[str, int]
    category_spend: Dict[str, float]
    product_counts: Dict[str, int]
    first_purchase: date
    last_purchase: date


@dataclass
class ProductProfile:
    product_id: str
    display_name: str
    category: str
    sub_category: str
    broad_category: str
    total_quantity: int
    total_revenue: float
    avg_price: float


@dataclass
class KnowledgeRecord:
    record_id: str
    text: str
    metadata: Dict
    vector: Dict[str, float]


class OrderAnalytics:
    def __init__(self, rows: List[Dict]):
        self.rows = rows
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.product_profiles: Dict[str, ProductProfile] = {}
        self.segment_breaks: Dict[str, float] = {}
        self.cooccurrence: Dict[Tuple[str, str], int] = {}
        self.unusual_events: List[Dict] = []
        self.knowledge_records: List[KnowledgeRecord] = []
        self.monthly_revenue: Dict[str, float] = {}
        self.category_hierarchy = self._build_category_hierarchy()
        self._build_profiles()
        self._build_segments()
        self._build_cooccurrence()
        self._build_temporal()
        self._detect_unusual_patterns()
        self._build_knowledge_base()

    def _build_category_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        hierarchy: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for row in self.rows:
            hierarchy[row["Broad_Category"]][row["Product_Category"]].add(row["Product_Sub_Category"])
        return {
            broad: {cat: sorted(list(subs)) for cat, subs in cats.items()}
            for broad, cats in hierarchy.items()
        }

    def _build_profiles(self) -> None:
        customer_rows: Dict[str, List[Dict]] = defaultdict(list)
        product_rows: Dict[str, List[Dict]] = defaultdict(list)
        for row in self.rows:
            customer_rows[row["Customer_ID"]].append(row)
            product_rows[row["Order_Lines_Product_ID"]].append(row)

        for cust_id, cust_rows in customer_rows.items():
            name = cust_rows[0]["Customer"]
            total_spend = sum(r["Subtotal"] for r in cust_rows)
            total_items = sum(r["Order_Quantity"] for r in cust_rows)
            total_orders = len({(r["Customer_ID"], r["Date"], r["Pricelist"]) for r in cust_rows})
            avg_ticket = total_spend / total_orders if total_orders else 0.0
            category_counts = Counter(r["Product_Category"] for r in cust_rows)
            category_spend = defaultdict(float)
            product_counts = Counter()
            for r in cust_rows:
                category_spend[r["Product_Category"]] += r["Subtotal"]
                product_counts[r["Product_Display_Name"]] += r["Order_Quantity"]
            first_purchase = min(r["Date"] for r in cust_rows)
            last_purchase = max(r["Date"] for r in cust_rows)
            self.customer_profiles[cust_id] = CustomerProfile(
                customer_id=cust_id,
                customer_name=name,
                total_spend=round(total_spend, 2),
                total_orders=total_orders,
                total_items=total_items,
                avg_ticket=round(avg_ticket, 2),
                category_counts=dict(category_counts),
                category_spend={k: round(v, 2) for k, v in category_spend.items()},
                product_counts=dict(product_counts),
                first_purchase=first_purchase,
                last_purchase=last_purchase,
            )

        for prod_id, prod_rows in product_rows.items():
            display_name = prod_rows[0]["Product_Display_Name"]
            category = prod_rows[0]["Product_Category"]
            sub_category = prod_rows[0]["Product_Sub_Category"]
            broad_category = prod_rows[0]["Broad_Category"]
            total_quantity = sum(r["Order_Quantity"] for r in prod_rows)
            total_revenue = sum(r["Subtotal"] for r in prod_rows)
            avg_price = statistics.mean(r["Order_Lines_Unit_Price"] for r in prod_rows)
            self.product_profiles[prod_id] = ProductProfile(
                product_id=prod_id,
                display_name=display_name,
                category=category,
                sub_category=sub_category,
                broad_category=broad_category,
                total_quantity=total_quantity,
                total_revenue=round(total_revenue, 2),
                avg_price=round(avg_price, 2),
            )

    def _build_segments(self) -> None:
        spends = [profile.total_spend for profile in self.customer_profiles.values()]
        if not spends:
            return
        high_cut = statistics.quantiles(spends, n=4)[-1]
        mid_cut = statistics.quantiles(spends, n=4)[1]
        self.segment_breaks = {"mid": mid_cut, "high": high_cut}

    def _build_cooccurrence(self) -> None:
        orders: Dict[Tuple[str, date, str], List[str]] = defaultdict(list)
        for row in self.rows:
            key = (row["Customer_ID"], row["Date"], row["Pricelist"])
            orders[key].append(row["Order_Lines_Product_ID"])
        pair_counts: Dict[Tuple[str, str], int] = Counter()
        for products in orders.values():
            unique_products = sorted(set(products))
            for i, prod_a in enumerate(unique_products):
                for prod_b in unique_products[i + 1 :]:
                    pair_counts[(prod_a, prod_b)] += 1
        self.cooccurrence = dict(pair_counts)

    def _build_temporal(self) -> None:
        monthly = defaultdict(float)
        for row in self.rows:
            key = f"{row['Date'].year}-{row['Date'].month:02d}"
            monthly[key] += row["Subtotal"]
        self.monthly_revenue = {k: round(v, 2) for k, v in monthly.items()}

    def _detect_unusual_patterns(self) -> None:
        if not self.rows:
            return
        latest_date = max(r["Date"] for r in self.rows)
        cutoff = date(latest_date.year, latest_date.month, 1)
        last_quarter_start = date(cutoff.year, cutoff.month - 2 if cutoff.month > 2 else 1 if cutoff.month == 3 else 1, 1)
        # Build baseline and recent category allocations per customer
        recent_window = {"Jun": 6, "Jul": 7, "Aug": 8}
        recent_months = { (latest_date.month - i - 1) % 12 + 1 for i in range(3) }
        customer_recent = defaultdict(lambda: defaultdict(float))
        customer_baseline = defaultdict(lambda: defaultdict(float))
        for row in self.rows:
            month = row["Date"].month
            bucket = customer_recent if month in recent_months else customer_baseline
            bucket[row["Customer_ID"]][row["Product_Category"]] += row["Subtotal"]
        events = []
        for cust_id, profile in self.customer_profiles.items():
            recent = customer_recent[cust_id]
            baseline = customer_baseline[cust_id]
            total_recent = sum(recent.values())
            total_base = sum(baseline.values())
            if total_recent < 1 or total_base < 1:
                continue
            for category, recent_spend in recent.items():
                base_spend = baseline.get(category, 0.0)
                if base_spend == 0 and recent_spend > 0:
                    change = math.inf
                else:
                    change = (recent_spend / total_recent) / (base_spend / total_base)
                if change >= 2 or change == math.inf:
                    events.append(
                        {
                            "customer_id": cust_id,
                            "customer_name": profile.customer_name,
                            "category": category,
                            "recent_share": round(recent_spend / total_recent, 2),
                            "previous_share": round((baseline.get(category, 0.0) / total_base) if total_base else 0.0, 2),
                            "change_ratio": round(change if change != math.inf else 999.0, 2),
                        }
                    )
        self.unusual_events = events

    def _build_knowledge_base(self) -> None:
        records: List[KnowledgeRecord] = []
        def add_record(record_id: str, text: str, metadata: Dict) -> None:
            records.append(
                KnowledgeRecord(
                    record_id=record_id,
                    text=text,
                    metadata=metadata,
                    vector=token_to_vector(text),
                )
            )

        for profile in self.customer_profiles.values():
            top_cats = sorted(profile.category_spend.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_cats:
                cat_text = ", ".join(f"{cat} (${spend:,.0f})" for cat, spend in top_cats)
                text = (
                    f"Customer {profile.customer_name} ({profile.customer_id}) spends ${profile.total_spend:,.0f} overall. "
                    f"Top categories: {cat_text}. Average ticket ${profile.avg_ticket:,.0f}."
                )
                add_record(
                    f"cust-{profile.customer_id}",
                    text,
                    {
                        "type": "customer_profile",
                        "customer_id": profile.customer_id,
                        "customer_name": profile.customer_name,
                    },
                )

        high_value_ids = self.high_value_customers()
        pairs = sorted(self.cooccurrence.items(), key=lambda x: x[1], reverse=True)[:50]
        for (prod_a, prod_b), count in pairs:
            prod_a_name = self.product_profiles[prod_a].display_name
            prod_b_name = self.product_profiles[prod_b].display_name
            text = (
                f"Products {prod_a_name} and {prod_b_name} appear together in {count} orders. "
                f"Categories: {self.product_profiles[prod_a].category} & {self.product_profiles[prod_b].category}."
            )
            add_record(
                f"pair-{prod_a}-{prod_b}",
                text,
                {
                    "type": "product_pair",
                    "product_a": prod_a,
                    "product_b": prod_b,
                    "count": count,
                },
            )

        for event in self.unusual_events:
            text = (
                f"Customer {event['customer_name']} ({event['customer_id']}) recently shifted towards {event['category']} "
                f"with share {event['recent_share']*100:.0f}% vs previous {event['previous_share']*100:.0f}%."
            )
            add_record(
                f"unusual-{event['customer_id']}-{event['category']}",
                text,
                {
                    "type": "unusual_pattern",
                    **event,
                },
            )

        self.knowledge_records = records

    def high_value_customers(self) -> List[str]:
        if not self.segment_breaks:
            return []
        threshold = self.segment_breaks["high"]
        return [cid for cid, profile in self.customer_profiles.items() if profile.total_spend >= threshold]

    def save_artifacts(self, directory: str = ARTIFACT_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        profiles_payload = {
            "customers": {cid: profile.__dict__ for cid, profile in self.customer_profiles.items()},
            "products": {pid: profile.__dict__ for pid, profile in self.product_profiles.items()},
            "segments": self.segment_breaks,
            "monthly_revenue": self.monthly_revenue,
            "category_hierarchy": self.category_hierarchy,
            "unusual_events": self.unusual_events,
        }
        with open(os.path.join(directory, "preprocessed_data.json"), "w", encoding="utf-8") as f:
            json.dump(profiles_payload, f, default=str, indent=2)
        kb_payload = [
            {
                "record_id": record.record_id,
                "text": record.text,
                "metadata": record.metadata,
            }
            for record in self.knowledge_records
        ]
        with open(os.path.join(directory, "knowledge_base.json"), "w", encoding="utf-8") as f:
            json.dump(kb_payload, f, indent=2)


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def token_to_vector(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    counts = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {token: count / norm for token, count in counts.items()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    dot = 0.0
    for token, value in vec_a.items():
        dot += value * vec_b.get(token, 0.0)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class QueryAgent:
    def __init__(self, analytics: OrderAnalytics):
        self.analytics = analytics

    def answer(self, question: str) -> Dict:
        lowered = question.lower()
        if "bought together" in lowered or "buy together" in lowered or "bundle" in lowered:
            return self._answer_cooccurrence(question)
        if "unusual" in lowered or "unique" in lowered or "anomal" in lowered:
            return self._answer_unusual(question)
        if "frequent" in lowered or "frequently" in lowered or "often" in lowered:
            return self._answer_frequent_purchase(question)
        return self._answer_via_retrieval(question)

    def _match_customer(self, question: str) -> Tuple[str, CustomerProfile]:
        lowered = question.lower()
        best_score = 0
        best_profile = None
        for profile in self.analytics.customer_profiles.values():
            name_tokens = tokenize(profile.customer_name)
            score = sum(1 for token in name_tokens if token in lowered)
            if profile.customer_id.lower() in lowered:
                score += 2
            if score > best_score:
                best_score = score
                best_profile = profile
        if best_profile:
            return best_profile.customer_id, best_profile
        raise ValueError("Customer not recognized in question.")

    def _match_category(self, question: str) -> str:
        lowered = question.lower()
        categories = set()
        for profile in self.analytics.product_profiles.values():
            categories.add(profile.category.lower())
            categories.add(profile.sub_category.lower())
            categories.add(profile.broad_category.lower())
        for category in sorted(categories, key=len, reverse=True):
            if category in lowered:
                return category
        return ""

    def _answer_frequent_purchase(self, question: str) -> Dict:
        try:
            cust_id, profile = self._match_customer(question)
        except ValueError:
            return {
                "answer": "I could not identify the customer in that question.",
                "reasoning": "Customer name or ID missing.",
            }
        category_term = self._match_category(question)
        response_lines: List[str] = []
        if category_term:
            matches = []
            for category, qty in profile.category_counts.items():
                if category_term in category.lower():
                    matches.append((category, qty))
            if not matches:
                for subcat in set(self.analytics.product_profiles[pid].sub_category for pid in self.analytics.product_profiles):
                    if category_term in subcat.lower():
                        # Map subcategory to products
                        matches = [(subcat, sum(q for prod, q in profile.product_counts.items() if category_term in prod.lower()))]
                        break
            if matches:
                category, _ = max(matches, key=lambda x: x[1])
                top_products = sorted(
                    (
                        (product, qty)
                        for product, qty in profile.product_counts.items()
                        if category_term in product.lower() or category_term in category.lower()
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                if not top_products:
                    top_products = sorted(profile.product_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                product_text = ", ".join(f"{name} ({qty})" for name, qty in top_products)
                response_lines.append(
                    f"{profile.customer_name} frequently buys {category} items. Top picks: {product_text}."
                )
            else:
                top_cat = max(profile.category_counts.items(), key=lambda x: x[1])
                response_lines.append(
                    f"No direct match for '{category_term}'. Their most purchased category is {top_cat[0]} with {top_cat[1]} items."
                )
        else:
            top_cat = max(profile.category_counts.items(), key=lambda x: x[1])
            response_lines.append(
                f"{profile.customer_name}'s dominant category is {top_cat[0]} with {top_cat[1]} items purchased."
            )
        response_lines.append(
            f"Total spend ${profile.total_spend:,.0f} across {profile.total_orders} orders (avg ticket ${profile.avg_ticket:,.0f})."
        )
        return {
            "answer": " ".join(response_lines),
            "reasoning": "Data aggregated from customer purchase history.",
            "supporting_facts": [
                {
                    "category_counts": profile.category_counts,
                    "top_products": sorted(profile.product_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                }
            ],
        }

    def _answer_cooccurrence(self, question: str) -> Dict:
        high_value = set(self.analytics.high_value_customers())
        pair_counts: Dict[Tuple[str, str], int] = Counter()
        orders: Dict[Tuple[str, date, str], List[str]] = defaultdict(list)
        for row in self.analytics.rows:
            if row["Customer_ID"] not in high_value:
                continue
            key = (row["Customer_ID"], row["Date"], row["Pricelist"])
            orders[key].append(row["Order_Lines_Product_ID"])
        for products in orders.values():
            unique_products = sorted(set(products))
            for i, prod_a in enumerate(unique_products):
                for prod_b in unique_products[i + 1 :]:
                    pair_counts[(prod_a, prod_b)] += 1
        if not pair_counts:
            return {
                "answer": "No high-value co-occurrence patterns detected.",
                "reasoning": "Co-occurrence matrix empty.",
            }
        top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = []
        for (prod_a, prod_b), count in top_pairs:
            profile_a = self.analytics.product_profiles[prod_a]
            profile_b = self.analytics.product_profiles[prod_b]
            lines.append(
                f"{profile_a.display_name} ({profile_a.category}) with {profile_b.display_name} ({profile_b.category}) - {count} joint orders."
            )
        return {
            "answer": "High-value customers tend to bundle: " + "; ".join(lines),
            "reasoning": "Pairs derived from high-value customer baskets.",
            "supporting_facts": [
                {"pair": pair, "count": count}
                for pair, count in top_pairs
            ],
        }

    def _answer_unusual(self, question: str) -> Dict:
        if not self.analytics.unusual_events:
            return {
                "answer": "No significant pattern shifts detected recently.",
                "reasoning": "Change thresholds not met.",
            }
        lines = []
        for event in sorted(self.analytics.unusual_events, key=lambda x: x["change_ratio"], reverse=True)[:5]:
            lines.append(
                f"{event['customer_name']} increased {event['category']} share to {event['recent_share']*100:.0f}% (was {event['previous_share']*100:.0f}%)."
            )
        return {
            "answer": "Recent anomalies: " + "; ".join(lines),
            "reasoning": "Comparison between last quarter and previous months.",
            "supporting_facts": self.analytics.unusual_events[:5],
        }

    def _answer_via_retrieval(self, question: str) -> Dict:
        query_vector = token_to_vector(question)
        scored: List[Tuple[float, KnowledgeRecord]] = []
        for record in self.analytics.knowledge_records:
            sim = cosine_similarity(query_vector, record.vector)
            if sim > 0:
                scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return {
                "answer": "I could not match that question to existing knowledge.",
                "reasoning": "No similar facts in the knowledge base.",
            }
        top_sim, top_record = scored[0]
        return {
            "answer": top_record.text,
            "reasoning": f"Retrieved top fact with cosine similarity {top_sim:.2f}.",
            "supporting_facts": [
                {"record_id": rec.record_id, "similarity": sim}
                for sim, rec in scored[:3]
            ],
        }


def build_agent(data_path: str = DATA_PATH) -> QueryAgent:
    rows = load_orders(data_path)
    analytics = OrderAnalytics(rows)
    analytics.save_artifacts()
    return QueryAgent(analytics)


def demo() -> List[Dict]:
    agent = build_agent()
    sample_questions = [
        "Which products does Customer CUST_015 frequently purchase in the Beverages category?",
        "Suggest products that are usually bought together by high-value customers.",
        "Identify unusual buying patterns in the last 3 months.",
        "What are the top categories for Customer_001?",
    ]
    responses = []
    for question in sample_questions:
        result = agent.answer(question)
        responses.append({"question": question, **result})
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "sample_responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
    return responses


if __name__ == "__main__":
    answers = demo()
    for entry in answers:
        print("Q:", entry["question"])
        print("A:", entry["answer"])
        print("Reasoning:", entry["reasoning"])
        print("-")
