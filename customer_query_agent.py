"""
Customer query agent for Dataset_product_orders.csv.
Performs preprocessing, builds lightweight embeddings, and answers natural language queries.
"""

import csv
import json
import math
import logging
import os
import statistics
import difflib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None

DATA_PATH = "Dataset_product_orders.csv"
ARTIFACT_DIR = "artifacts"


def load_env_from_file(path: str = ".env") -> None:
    """Load environment variables from a local .env file if present."""
    env_path = Path(path)
    candidates = []
    if env_path.is_absolute():
        candidates.append(env_path)
    else:
        candidates.append(Path.cwd() / env_path)
        candidates.append(Path(__file__).resolve().parent / env_path)
    for candidate in candidates:
        if candidate.exists():
            env_path = candidate
            break
    else:
        return
    try:
        with env_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value
    except OSError:
        return


BENCHMARK_CASES = [
    {
        "question": "What is the overall spend and top categories for CUST_001?",
        "expected_prefix": "cust-CUST_001",
        "description": "Customer profile retrieval",
    },
    {
        "question": "Highlight any notable product bundle fact.",
        "expected_prefix": "pair-",
        "description": "Product pair retrieval",
    },
    {
        "question": "Who recently shifted category spend patterns?",
        "expected_prefix": "unusual-",
        "description": "Anomaly fact retrieval",
    },
    {
        "question": "What is the overall spend and top categories for CUST_015?",
        "expected_prefix": "cust-CUST_015",
        "description": "Customer profile retrieval",
    },
    {
        "question": "Which products do high-value shoppers bundle with Product 77?",
        "expected_prefix": "pair-",
        "description": "Product pair retrieval",
    },
    {
        "question": "Call out any unusual shift for Bonnie Garrett.",
        "expected_prefix": "unusual-CUST_004",
        "description": "Anomaly fact retrieval",
    },
]

load_env_from_file()

LOGGER = logging.getLogger(__name__)

def _clean_row(raw_row: Dict[str, Optional[str]], row_number: int) -> Optional[Dict[str, object]]:
    cleaned: Dict[str, object] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        cleaned[key] = value

    required_fields = [
        "Customer_ID",
        "Order_Lines_Product_ID",
        "Order_Quantity",
        "Order_Lines_Unit_Price",
        "Subtotal",
        "Total_OrderQuantity",
        "Total_Subtotal",
        "Date",
    ]

    for field in required_fields:
        if not cleaned.get(field):
            LOGGER.warning("Skipping row %s due to missing '%s'", row_number, field)
            return None

    try:
        cleaned["Order_Quantity"] = int(cleaned["Order_Quantity"])
        cleaned["Order_Lines_Unit_Price"] = float(cleaned["Order_Lines_Unit_Price"])
        cleaned["Subtotal"] = float(cleaned["Subtotal"])
        cleaned["Total_OrderQuantity"] = int(cleaned["Total_OrderQuantity"])
        cleaned["Total_Subtotal"] = float(cleaned["Total_Subtotal"])
        cleaned["Date"] = datetime.strptime(str(cleaned["Date"]), "%Y-%m-%d").date()
        cleaned["Month"] = date(cleaned["Date"].year, cleaned["Date"].month, 1)
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Skipping row %s due to parse error: %s", row_number, exc)
        return None

    return cleaned


def load_orders(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, raw_row in enumerate(reader, start=2):  # account for header row
            cleaned = _clean_row(raw_row, idx)
            if cleaned is None:
                continue
            rows.append(cleaned)
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
    unique_customers: int


@dataclass
class KnowledgeRecord:
    record_id: str
    text: str
    metadata: Dict
    vector: Dict[str, float]


class GeminiClient:
    """Thin wrapper around the modern Google Gemini API client."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if google_genai is None:
            raise ImportError("The google-genai client library is not available.")
        self.model_name = model
        self._logger = logging.getLogger(__name__)
        self._client = google_genai.Client(api_key=api_key)

    @classmethod
    def from_env(cls) -> Optional["GeminiClient"]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        if google_genai is None:
            return None
        try:
            return cls(api_key=api_key)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).warning("Failed to initialize Gemini client: %s", exc)
            return None

    def _build_config(self, response_mime_type: Optional[str]) -> Optional[object]:
        if google_genai_types is None:
            return None
        cfg_kwargs = {}
        if response_mime_type:
            cfg_kwargs["response_mime_type"] = response_mime_type
        thinking_config = None
        if hasattr(google_genai_types, "ThinkingConfig"):
            thinking_config = google_genai_types.ThinkingConfig(thinking_budget=0)
        if thinking_config is not None:
            cfg_kwargs["thinking_config"] = thinking_config
        if not cfg_kwargs:
            return None
        if hasattr(google_genai_types, "GenerateContentConfig"):
            return google_genai_types.GenerateContentConfig(**cfg_kwargs)
        return None

    def _extract_text(self, response: object) -> Optional[str]:
        text_value = getattr(response, "text", None)
        if text_value:
            return text_value.strip()
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            part_iter = []
            content = getattr(candidate, "content", None)
            if hasattr(content, "parts"):
                part_iter = content.parts
            elif content:
                part_iter = content
            else:
                part_iter = getattr(candidate, "parts", [])
            for part in part_iter:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text.strip()
        return None

    def generate(self, prompt: str, response_mime_type: Optional[str] = None) -> Optional[str]:
        try:
            kwargs = {
                "model": self.model_name,
                "contents": prompt,
            }
            config = self._build_config(response_mime_type)
            if config is not None:
                kwargs["config"] = config
            response = self._client.models.generate_content(**kwargs)
        except Exception as exc:  # pragma: no cover - network/runtime issues
            self._logger.warning("Gemini generation failed: %s", exc)
            return None
        return self._extract_text(response)

    def generate_json(self, prompt: str) -> Optional[Dict]:
        raw = self.generate(prompt, response_mime_type="application/json")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._logger.warning("Gemini JSON decode failed: %s", raw)
            return None


class OrderAnalytics:
    """Lightweight analytics layer that prepares customer, product, and temporal summaries."""

    def __init__(self, rows: List[Dict]):
        self.rows = rows
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.product_profiles: Dict[str, ProductProfile] = {}
        self.segment_breaks: Dict[str, float] = {}
        self.cooccurrence: Dict[Tuple[str, str], int] = {}
        self.unusual_events: List[Dict] = []
        self.knowledge_records: List[KnowledgeRecord] = []
        self.monthly_revenue: Dict[str, float] = {}
        self.dataset_overview: Dict[str, object] = {}
        self.numeric_summary: Dict[str, Dict[str, float]] = {}
        self.broad_category_summary: List[Dict] = []
        self.category_summary: List[Dict] = []
        self.product_rollup: List[Dict] = []
        self.customer_rollup: List[Dict] = []
        self.pricelist_summary: Dict[str, Dict[str, float]] = {}
        self.daily_revenue_series: List[Dict] = []
        self.monthly_category_series: List[Dict] = []
        self.top_cooccurrences: List[Dict] = []
        self.category_hierarchy = self._build_category_hierarchy()
        self._build_profiles()
        self._build_segments()
        self._build_cooccurrence()
        self._build_temporal()
        self._detect_unusual_patterns()
        self._build_knowledge_base()
        self._build_alignment_views()

    def _build_category_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        hierarchy: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for row in self.rows:
            hierarchy[row["Broad_Category"]][row["Product_Category"]].add(row["Product_Sub_Category"])
        return {
            broad: {cat: sorted(list(subs)) for cat, subs in cats.items()}
            for broad, cats in hierarchy.items()
        }

    def _build_profiles(self) -> None:
        """Aggregate raw rows into customer and product profile objects."""
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
            unique_customers = len({r["Customer_ID"] for r in prod_rows})
            self.product_profiles[prod_id] = ProductProfile(
                product_id=prod_id,
                display_name=display_name,
                category=category,
                sub_category=sub_category,
                broad_category=broad_category,
                total_quantity=total_quantity,
                total_revenue=round(total_revenue, 2),
                avg_price=round(avg_price, 2),
                unique_customers=unique_customers,
            )

    def _build_segments(self) -> None:
        """Compute spend-based thresholds used for tiering customers."""
        spends = [profile.total_spend for profile in self.customer_profiles.values()]
        if not spends:
            return
        high_cut = statistics.quantiles(spends, n=4)[-1]
        mid_cut = statistics.quantiles(spends, n=4)[1]
        self.segment_breaks = {"mid": mid_cut, "high": high_cut}

    def _build_cooccurrence(self) -> None:
        """Approximate basket co-occurrence counts for recommendation prompts."""
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
        """Summarise revenue at a monthly cadence for trend analysis."""
        monthly = defaultdict(float)
        for row in self.rows:
            key = f"{row['Date'].year}-{row['Date'].month:02d}"
            monthly[key] += row["Subtotal"]
        self.monthly_revenue = {k: round(v, 2) for k, v in monthly.items()}

    def _detect_unusual_patterns(self) -> None:
        """Flag customers whose category share shifted sharply in recent months."""
        if not self.rows:
            return
        latest_date = max(r["Date"] for r in self.rows)

        def _previous_months(anchor: date, count: int) -> Set[Tuple[int, int]]:
            year, month = anchor.year, anchor.month
            results: Set[Tuple[int, int]] = set()
            for _ in range(count):
                results.add((year, month))
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
            return results

        recent_months = _previous_months(latest_date, 3)
        customer_recent = defaultdict(lambda: defaultdict(float))
        customer_baseline = defaultdict(lambda: defaultdict(float))
        for row in self.rows:
            month_key = (row["Date"].year, row["Date"].month)
            bucket = customer_recent if month_key in recent_months else customer_baseline
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
        """Serialise factual snippets that feed the retrieval component."""
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

    def _build_alignment_views(self) -> None:
        """Prepare helper summaries aligned with the exploratory analysis."""
        if not self.rows:
            return
        self.dataset_overview = self._compute_overview()
        self.numeric_summary = self._compute_numeric_summary()
        self.broad_category_summary = self._compute_broad_category_summary()
        self.category_summary = self._compute_category_summary()
        self.product_rollup = self._compute_product_rollup()
        self.customer_rollup = self._compute_customer_rollup()
        self.pricelist_summary = self._compute_pricelist_summary(self.customer_rollup)
        self.daily_revenue_series = self._compute_daily_revenue_series()
        self.monthly_category_series = self._compute_monthly_category_series()
        self.top_cooccurrences = self._compute_top_cooccurrences()

    def _compute_overview(self) -> Dict[str, object]:
        dates = [row['Date'] for row in self.rows]
        return {
            'rows': len(self.rows),
            'columns': len(self.rows[0]) if self.rows else 0,
            'date_range': {
                'min': min(dates).isoformat(),
                'max': max(dates).isoformat(),
            },
            'unique_customers': len(self.customer_profiles),
            'unique_products': len(self.product_profiles),
            'broad_categories': len({row['Broad_Category'] for row in self.rows}),
            'pricelists': sorted({row['Pricelist'] for row in self.rows}),
        }

    def _compute_numeric_summary(self) -> Dict[str, Dict[str, float]]:
        numeric_fields = [
            'Order_Lines_Unit_Price',
            'Order_Quantity',
            'Subtotal',
            'Total_OrderQuantity',
            'Total_Subtotal',
        ]
        summary: Dict[str, Dict[str, float]] = {}
        for field in numeric_fields:
            values = [float(row[field]) for row in self.rows]
            values.sort()
            summary[field] = {
                'min': round(values[0], 2),
                'max': round(values[-1], 2),
                'mean': round(statistics.mean(values), 2),
                'median': round(statistics.median(values), 2),
                'p10': round(self._quantile(values, 0.10), 2),
                'p25': round(self._quantile(values, 0.25), 2),
                'p50': round(self._quantile(values, 0.50), 2),
                'p75': round(self._quantile(values, 0.75), 2),
                'p90': round(self._quantile(values, 0.90), 2),
                'p95': round(self._quantile(values, 0.95), 2),
            }
        return summary

    def _compute_broad_category_summary(self) -> List[Dict]:
        summary: Dict[str, Dict[str, object]] = {}
        total_revenue = 0.0
        for row in self.rows:
            broad = row['Broad_Category']
            record = summary.setdefault(
                broad,
                {
                    'broad_category': broad,
                    'total_revenue': 0.0,
                    'total_quantity': 0,
                    'unique_products': set(),
                },
            )
            record['total_revenue'] += row['Subtotal']
            record['total_quantity'] += row['Order_Quantity']
            record['unique_products'].add(row['Product_Display_Name'])
            total_revenue += row['Subtotal']
        output: List[Dict] = []
        for record in summary.values():
            revenue = record['total_revenue']
            output.append(
                {
                    'broad_category': record['broad_category'],
                    'total_revenue': round(revenue, 2),
                    'revenue_pct': round(revenue / total_revenue, 4) if total_revenue else 0.0,
                    'total_quantity': record['total_quantity'],
                    'unique_products': len(record['unique_products']),
                }
            )
        output.sort(key=lambda x: x['total_revenue'], reverse=True)
        return output

    def _compute_category_summary(self) -> List[Dict]:
        summary: Dict[Tuple[str, str], Dict[str, object]] = {}
        for row in self.rows:
            key = (row['Broad_Category'], row['Product_Category'])
            record = summary.setdefault(
                key,
                {
                    'broad_category': row['Broad_Category'],
                    'product_category': row['Product_Category'],
                    'total_revenue': 0.0,
                    'total_quantity': 0,
                    'unique_products': set(),
                },
            )
            record['total_revenue'] += row['Subtotal']
            record['total_quantity'] += row['Order_Quantity']
            record['unique_products'].add(row['Product_Display_Name'])
        output: List[Dict] = []
        for record in summary.values():
            output.append(
                {
                    'broad_category': record['broad_category'],
                    'product_category': record['product_category'],
                    'total_revenue': round(record['total_revenue'], 2),
                    'total_quantity': record['total_quantity'],
                    'unique_products': len(record['unique_products']),
                }
            )
        output.sort(key=lambda x: x['total_revenue'], reverse=True)
        return output

    def _compute_product_rollup(self) -> List[Dict]:
        items: List[Dict] = []
        for profile in self.product_profiles.values():
            items.append(
                {
                    'product_id': profile.product_id,
                    'product_name': profile.display_name,
                    'product_category': profile.category,
                    'product_sub_category': profile.sub_category,
                    'broad_category': profile.broad_category,
                    'total_revenue': profile.total_revenue,
                    'total_quantity': profile.total_quantity,
                    'average_unit_price': round(profile.avg_price, 2),
                    'unique_customers': profile.unique_customers,
                }
            )
        items.sort(key=lambda x: x['total_revenue'], reverse=True)
        return items

    def _compute_customer_rollup(self) -> List[Dict]:
        rollup: Dict[Tuple[str, str], Dict[str, object]] = {}
        for row in self.rows:
            key = (row['Customer_ID'], row['Pricelist'])
            record = rollup.setdefault(
                key,
                {
                    'customer_id': row['Customer_ID'],
                    'customer_name': row['Customer'],
                    'pricelist': row['Pricelist'],
                    'total_revenue': 0.0,
                    'total_quantity': 0,
                    'unique_products': set(),
                    'first_purchase': row['Date'],
                    'last_purchase': row['Date'],
                },
            )
            record['total_revenue'] += row['Subtotal']
            record['total_quantity'] += row['Order_Quantity']
            record['unique_products'].add(row['Product_Display_Name'])
            record['first_purchase'] = min(record['first_purchase'], row['Date'])
            record['last_purchase'] = max(record['last_purchase'], row['Date'])
        output: List[Dict] = []
        for record in rollup.values():
            output.append(
                {
                    'customer_id': record['customer_id'],
                    'customer_name': record['customer_name'],
                    'pricelist': record['pricelist'],
                    'total_revenue': round(record['total_revenue'], 2),
                    'total_quantity': record['total_quantity'],
                    'unique_products': len(record['unique_products']),
                    'first_purchase': record['first_purchase'],
                    'last_purchase': record['last_purchase'],
                    'tenure_days': (record['last_purchase'] - record['first_purchase']).days,
                }
            )
        output.sort(key=lambda x: x['total_revenue'], reverse=True)
        return output

    def _compute_pricelist_summary(self, customer_rollup: List[Dict]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for record in customer_rollup:
            grouped[record['pricelist']].append(record)
        for pricelist, records in grouped.items():
            total_revenue = sum(r['total_revenue'] for r in records)
            total_quantity = sum(r['total_quantity'] for r in records)
            customers = len(records)
            revenue_values = [r['total_revenue'] for r in records]
            avg_rev = total_revenue / customers if customers else 0.0
            median_rev = statistics.median(revenue_values) if revenue_values else 0.0
            summary[pricelist] = {
                'customers': customers,
                'total_revenue': round(total_revenue, 2),
                'total_quantity': total_quantity,
                'avg_revenue_per_customer': round(avg_rev, 2),
                'median_revenue_per_customer': round(median_rev, 2),
            }
        return summary

    def _compute_daily_revenue_series(self) -> List[Dict]:
        daily: Dict[date, float] = defaultdict(float)
        for row in self.rows:
            daily[row['Date']] += row['Subtotal']
        series: List[Dict] = []
        window: deque = deque()
        rolling_sum = 0.0
        for current_date in sorted(daily):
            value = round(daily[current_date], 2)
            window.append((current_date, value))
            rolling_sum += value
            if len(window) > 7:
                _, removed = window.popleft()
                rolling_sum -= removed
            rolling_avg = round(rolling_sum / len(window), 2)
            series.append(
                {
                    'date': current_date.isoformat(),
                    'revenue': value,
                    'rolling_7d': rolling_avg,
                }
            )
        return series

    def _compute_monthly_category_series(self) -> List[Dict]:
        monthly: Dict[date, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in self.rows:
            key = date(row['Date'].year, row['Date'].month, 1)
            monthly[key][row['Broad_Category']] += row['Subtotal']
        series: List[Dict] = []
        for month_key in sorted(monthly):
            for category, revenue in monthly[month_key].items():
                series.append(
                    {
                        'month': month_key.isoformat(),
                        'broad_category': category,
                        'revenue': round(revenue, 2),
                    }
                )
        return series

    def _compute_top_cooccurrences(self, limit: int = 15) -> List[Dict]:
        records: List[Dict] = []
        for (prod_a, prod_b), count in sorted(self.cooccurrence.items(), key=lambda x: x[1], reverse=True)[:limit]:
            profile_a = self.product_profiles.get(prod_a)
            profile_b = self.product_profiles.get(prod_b)
            records.append(
                {
                    'product_a_id': prod_a,
                    'product_a_name': profile_a.display_name if profile_a else prod_a,
                    'product_b_id': prod_b,
                    'product_b_name': profile_b.display_name if profile_b else prod_b,
                    'cooccurrences': count,
                }
            )
        return records

    @staticmethod
    def _quantile(sorted_values: List[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        if q <= 0:
            return float(sorted_values[0])
        if q >= 1:
            return float(sorted_values[-1])
        pos = (len(sorted_values) - 1) * q
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(sorted_values[int(pos)])
        lower_value = sorted_values[lower]
        upper_value = sorted_values[upper]
        return float(lower_value + (upper_value - lower_value) * (pos - lower))

    def retrieve_records(self, query: str, top_k: int = 5) -> List[Tuple[float, KnowledgeRecord]]:
        """Return the top matching knowledge base entries for a question."""
        if not self.knowledge_records:
            return []
        query_vector = token_to_vector(query)
        scored: List[Tuple[float, KnowledgeRecord]] = []
        for record in self.knowledge_records:
            sim = cosine_similarity(query_vector, record.vector)
            if sim > 0:
                scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def high_value_customers(self) -> List[str]:
        """List customer IDs that fall into the highest spend tier."""
        if not self.segment_breaks:
            return []
        threshold = self.segment_breaks["high"]
        return [cid for cid, profile in self.customer_profiles.items() if profile.total_spend >= threshold]

    def save_artifacts(self, directory: str = ARTIFACT_DIR) -> None:
        """Persist key analytics structures and knowledge base files to disk."""
        os.makedirs(directory, exist_ok=True)
        profiles_payload = {
            "customers": {cid: profile.__dict__ for cid, profile in self.customer_profiles.items()},
            "products": {pid: profile.__dict__ for pid, profile in self.product_profiles.items()},
            "segments": self.segment_breaks,
            "monthly_revenue": self.monthly_revenue,
            "category_hierarchy": self.category_hierarchy,
            "unusual_events": self.unusual_events,
        }
        profiles_payload["eda_alignment"] = {
            "overview": self.dataset_overview,
            "numeric_summary": self.numeric_summary,
            "broad_category_summary": self.broad_category_summary,
            "category_summary_top15": self.category_summary[:15],
            "product_rollup_top15": self.product_rollup[:15],
            "customer_rollup_top15": self.customer_rollup[:15],
            "pricelist_summary": self.pricelist_summary,
            "daily_revenue_series": self.daily_revenue_series,
            "monthly_category_series": self.monthly_category_series,
            "top_cooccurrences": self.top_cooccurrences,
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


def run_retrieval_benchmark(analytics: "OrderAnalytics", directory: str = ARTIFACT_DIR) -> Dict[str, object]:
    if not BENCHMARK_CASES:
        return {
            "total_cases": 0,
            "hits": 0,
            "precision_at_1": 0.0,
            "cases": [],
        }
    os.makedirs(directory, exist_ok=True)
    cases: List[Dict[str, object]] = []
    hits = 0
    for case in BENCHMARK_CASES:
        scored = analytics.retrieve_records(case["question"], top_k=3)
        if scored:
            top_score, top_record = scored[0]
            top_record_id = top_record.record_id
            similarity = round(top_score, 3)
            hit = bool(top_record_id and top_record_id.startswith(case["expected_prefix"]))
        else:
            top_record_id = None
            similarity = 0.0
            hit = False
        if hit:
            hits += 1
        cases.append(
            {
                "question": case["question"],
                "description": case.get("description"),
                "top_record_id": top_record_id,
                "top_similarity": similarity,
                "hit": hit,
            }
        )
    precision = hits / len(BENCHMARK_CASES) if BENCHMARK_CASES else 0.0
    summary = {
        "total_cases": len(BENCHMARK_CASES),
        "hits": hits,
        "precision_at_1": round(precision, 3),
        "cases": cases,
    }
    with open(os.path.join(directory, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


class QueryAgent:
    def __init__(self, analytics: OrderAnalytics):
        self.analytics = analytics
        self.llm_client = GeminiClient.from_env()

    def answer(self, question: str) -> Dict:
        """Primary entry point for answering natural language customer questions."""
        intent = self._classify_intent(question)
        analysis_trace = [
            {"step": "intent_classification", "intent": intent},
        ]
        retrieved = self.analytics.retrieve_records(question, top_k=5)
        analysis_trace.append({"step": "retrieval", "hits": len(retrieved)})
        if intent == "cooccurrence":
            response = self._answer_cooccurrence(question)
        elif intent == "unusual":
            response = self._answer_unusual(question)
        elif intent == "frequent_purchase":
            response = self._answer_frequent_purchase(question)
        else:
            response = self._answer_via_retrieval(question, scored=retrieved)
        if retrieved:
            response["retrieved_records"] = [
                {
                    "record_id": record.record_id,
                    "similarity": round(score, 3),
                    "text": record.text,
                    "metadata": record.metadata,
                }
                for score, record in retrieved
            ]
        response["analysis_trace"] = analysis_trace
        if self.llm_client:
            llm_answer = self._synthesise_with_llm(question, response, retrieved, intent)
            if llm_answer:
                if response.get("answer"):
                    response.setdefault("answer_raw", response["answer"])
                response["answer"] = llm_answer
                if response.get("reasoning"):
                    response["reasoning"] = response["reasoning"].rstrip('.') + '. Narrative composed with Gemini.'
                else:
                    response["reasoning"] = 'Narrative composed with Gemini.'
                response["llm_model"] = self.llm_client.model_name
                analysis_trace.append({"step": "llm_synthesis", "model": self.llm_client.model_name})
                response["analysis_trace"] = analysis_trace
        return response

    def _rule_based_intent(self, question: str) -> str:
        lowered = question.lower()
        if "bought together" in lowered or "buy together" in lowered or "bundle" in lowered:
            return "cooccurrence"
        if "unusual" in lowered or "unique" in lowered or "anomal" in lowered:
            return "unusual"
        if "frequent" in lowered or "frequently" in lowered or "often" in lowered:
            return "frequent_purchase"
        return "retrieval"

    def _classify_intent(self, question: str) -> str:
        """Route questions to specialised handlers or the retrieval fallback."""
        print(f"Classifying intent for question: {question}")
        print(f"Using LLM client: {self.llm_client is not None}")
        if not self.llm_client:
            return self._rule_based_intent(question)
        prompt = (
            "Classify the intent of the following customer analytics question into one of ['frequent_purchase', 'cooccurrence', 'unusual', 'retrieval'].\n"
            "Respond with a JSON object {\"intent\": <label>} without additional text.\n"
            f"Question: {question}\n"
        )
        result = self.llm_client.generate_json(prompt)
        if isinstance(result, dict):
            intent = result.get("intent")
            if intent in {"frequent_purchase", "cooccurrence", "unusual", "retrieval"}:
                return intent
        return self._rule_based_intent(question)

    def _synthesise_with_llm(
        self,
        question: str,
        base_response: Dict,
        retrieved: List[Tuple[float, KnowledgeRecord]],
        intent: str,
    ) -> Optional[str]:
        """Ask Gemini to polish the heuristic answer when the API is available."""
        if not self.llm_client:
            return None
        payload = {
            "intent": intent,
            "question": question,
            "heuristic_answer": base_response.get("answer"),
            "supporting_facts": base_response.get("supporting_facts", []),
            "retrieved_records": [
                {
                    "record_id": record.record_id,
                    "similarity": round(score, 3),
                    "text": record.text,
                    "metadata": record.metadata,
                }
                for score, record in retrieved
            ],
            "dataset_overview": self.analytics.dataset_overview,
            "top_broad_categories": self.analytics.broad_category_summary[:5],
            "top_products": self.analytics.product_rollup[:10],
            "top_customers": self.analytics.customer_rollup[:10],
        }
        context = json.dumps(payload, indent=2, default=str)
        prompt = (
            "You are an analytical assistant. Use ONLY the provided context to answer the question.\n"
            "Your job is to refine the heuristic answer without contradicting the provided metrics or supporting facts.\n"
            "Never invent numbers, and prefer saying what is known over speculation.\n"
            "If data is missing, acknowledge the gap and suggest next steps based on the context.\n"
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            "Craft a concise, well-structured answer grounded in these numbers."
        )
        return self.llm_client.generate(prompt)

    def _match_customer(self, question: str) -> Tuple[str, CustomerProfile]:
        """Best-effort match between a question and a known customer profile."""
        lowered = question.lower()
        best_score = 0.0
        best_profile: Optional[CustomerProfile] = None
        for profile in self.analytics.customer_profiles.values():
            name_tokens = tokenize(profile.customer_name)
            token_hits = sum(1 for token in name_tokens if token in lowered)
            name_ratio = difflib.SequenceMatcher(None, lowered, profile.customer_name.lower()).ratio()
            id_ratio = difflib.SequenceMatcher(None, lowered, profile.customer_id.lower()).ratio()
            score = max(name_ratio, id_ratio) + token_hits * 0.1
            if profile.customer_id.lower() in lowered:
                score += 0.5
            if score > best_score:
                best_score = score
                best_profile = profile
        if best_profile and best_score >= 0.35:
            return best_profile.customer_id, best_profile
        if best_profile:  # fall back to best guess even if weak match
            return best_profile.customer_id, best_profile
        raise ValueError("Customer not recognized in question.")

    def _match_category(self, question: str) -> str:
        """Resolve a category mention inside the question to our taxonomy labels."""
        lowered = question.lower()
        candidates: Dict[str, str] = {}
        for profile in self.analytics.product_profiles.values():
            candidates.setdefault(profile.category.lower(), profile.category)
            candidates.setdefault(profile.sub_category.lower(), profile.sub_category)
            candidates.setdefault(profile.broad_category.lower(), profile.broad_category)
        direct_hits = [key for key in candidates if key in lowered]
        if direct_hits:
            best = max(direct_hits, key=len)
            return candidates[best]
        best_match = None
        best_score = 0.0
        for key, original in candidates.items():
            score = difflib.SequenceMatcher(None, lowered, key).ratio()
            if score > best_score:
                best_score = score
                best_match = original
        return best_match or ""

    def _answer_frequent_purchase(self, question: str) -> Dict:
        """Summarise what a customer buys most often, optionally scoped to a category."""
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
            category_key = category_term.lower()
            matches = []
            for category, qty in profile.category_counts.items():
                if category_key in category.lower():
                    matches.append((category, qty))
            if not matches:
                for subcat in set(self.analytics.product_profiles[pid].sub_category for pid in self.analytics.product_profiles):
                    if category_key in subcat.lower():
                        matches = [
                            (
                                subcat,
                                sum(
                                    q
                                    for prod, q in profile.product_counts.items()
                                    if category_key in prod.lower()
                                ),
                            )
                        ]
                        break
            if matches:
                category, _ = max(matches, key=lambda x: x[1])
                top_products = sorted(
                    (
                        (product, qty)
                        for product, qty in profile.product_counts.items()
                        if category_key in product.lower() or category_key in category.lower()
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
        """Highlight high-value customer basket pairings."""
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
        """Report recent category share shifts marked as unusual."""
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

    def _answer_via_retrieval(self, question: str, scored: Optional[List[Tuple[float, KnowledgeRecord]]] = None) -> Dict:
        """Fallback path that returns the top retrieved knowledge record."""
        scored = scored or self.analytics.retrieve_records(question, top_k=5)
        if not scored:
            return self._fallback_overview(question)
        top_sim, top_record = scored[0]
        return {
            "answer": top_record.text,
            "reasoning": f"Retrieved top fact with cosine similarity {top_sim:.2f}.",
            "supporting_facts": [
                {"record_id": rec.record_id, "similarity": sim}
                for sim, rec in scored[:3]
            ],
        }

    def _fallback_overview(self, question: str) -> Dict:
        """Provide baseline dataset context when retrieval does not help."""
        overview = self.analytics.dataset_overview or {}
        top_categories = self.analytics.broad_category_summary[:3]
        lines = []
        if overview:
            lines.append(
                f"Dataset covers {overview.get('rows', 0)} rows across {overview.get('columns', 0)} columns from {overview.get('date_range', {}).get('min', '?')} to {overview.get('date_range', {}).get('max', '?')}."
            )
            pricelists = overview.get('pricelists', [])
            if pricelists:
                lines.append(
                    f"There are {overview.get('unique_customers', 0)} customers and {overview.get('unique_products', 0)} products across pricelists {', '.join(pricelists)}."
                )
            else:
                lines.append(
                    f"There are {overview.get('unique_customers', 0)} customers and {overview.get('unique_products', 0)} products."
                )
        if top_categories:
            category_line = '; '.join(
                f"{entry['broad_category']} (${entry['total_revenue']:,})" for entry in top_categories
            )
            lines.append(f"Top revenue categories: {category_line}.")
        if not lines:
            lines.append("No matching knowledge found; dataset overview unavailable.")
        return {
            "answer": " ".join(lines),
            "reasoning": "Fallback to dataset overview because no close knowledge records matched.",
            "supporting_facts": [
                {"overview": overview},
                {"top_categories": top_categories},
                {"note": "Question was: " + question},
            ],
        }


def build_agent(data_path: str = DATA_PATH) -> QueryAgent:
    """Load data, build analytics, and return a ready-to-use query agent."""
    rows = load_orders(data_path)
    analytics = OrderAnalytics(rows)
    analytics.save_artifacts()
    return QueryAgent(analytics)


def demo() -> List[Dict]:
    """Run a demo sequence that prints sample answers and refreshes artifacts."""
    agent = build_agent()
    sample_questions = [
        "Which products does Customer CUST_015 frequently purchase in the Beverages category?",
        "Suggest products that are usually bought together by high-value customers.",
        "Identify unusual buying patterns in the last 3 months.",
        "What are the top categories for Customer_001?",
        "who purchase the most"
    ]
    responses = []
    for question in sample_questions:
        result = agent.answer(question)
        responses.append({"question": question, **result})
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "sample_responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
    benchmark_summary = run_retrieval_benchmark(agent.analytics, directory=ARTIFACT_DIR)
    print(
        f"Retrieval benchmark precision@1: {benchmark_summary['precision_at_1']:.3f} "
        f"({benchmark_summary['hits']} of {benchmark_summary['total_cases']})"
    )
    return responses


if __name__ == "__main__":
    answers = demo()
    for entry in answers:
        print("Q:", entry["question"])
        print("A:", entry["answer"])
        print("Reasoning:", entry["reasoning"])
        print("-")
