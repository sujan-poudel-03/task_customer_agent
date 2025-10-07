"""Bonus challenge utilities for dynamic recommendations and multi-customer reasoning."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from customer_query_agent import DATA_PATH, OrderAnalytics, load_orders


def _normalise(text: str) -> str:
    """Lowercase the text and collapse non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class QueryContext:
    customers: Set[str]
    categories: Set[str]
    broad_categories: Set[str]
    sub_categories: Set[str]


class BonusAdvisor:
    """Provides dynamic recommendations and cross-customer reasoning."""

    def __init__(self, analytics: OrderAnalytics) -> None:
        self.analytics = analytics
        self._customer_aliases: Dict[str, str] = {}
        self._category_aliases: Dict[str, str] = {}
        self._broad_aliases: Dict[str, str] = {}
        self._sub_aliases: Dict[str, Tuple[str, str]] = {}
        self._category_to_broad: Dict[str, str] = {}
        self._build_lookups()

    @classmethod
    def from_dataset(cls, path: str = DATA_PATH) -> "BonusAdvisor":
        rows = load_orders(path)
        analytics = OrderAnalytics(rows)
        return cls(analytics)

    def _build_lookups(self) -> None:
        for profile in self.analytics.customer_profiles.values():
            self._customer_aliases[_normalise(profile.customer_id)] = profile.customer_id
            self._customer_aliases[_normalise(profile.customer_name)] = profile.customer_id

        for product in self.analytics.product_profiles.values():
            normal_category = _normalise(product.category)
            normal_broad = _normalise(product.broad_category)
            normal_sub = _normalise(product.sub_category)
            self._category_aliases[normal_category] = product.category
            self._broad_aliases[normal_broad] = product.broad_category
            self._sub_aliases[normal_sub] = (product.sub_category, product.category)
            self._category_to_broad[product.category] = product.broad_category

        for broad, categories in self.analytics.category_hierarchy.items():
            self._broad_aliases.setdefault(_normalise(broad), broad)
            for category, subs in categories.items():
                self._category_aliases.setdefault(_normalise(category), category)
                self._category_to_broad.setdefault(category, broad)
                for sub in subs:
                    self._sub_aliases.setdefault(_normalise(sub), (sub, category))

    def _extract_context(self, question: str) -> QueryContext:
        normalized = _normalise(question)
        customers: Set[str] = set()
        categories: Set[str] = set()
        broad_categories: Set[str] = set()
        sub_categories: Set[str] = set()

        for match in re.findall(r"cust[_-]\d+", question, flags=re.IGNORECASE):
            key = match.replace("-", "_").upper()
            if key in self.analytics.customer_profiles:
                customers.add(key)

        for alias, cust_id in self._customer_aliases.items():
            if alias and alias in normalized:
                customers.add(cust_id)

        for alias, category in self._category_aliases.items():
            if alias and alias in normalized:
                categories.add(category)

        for alias, broad in self._broad_aliases.items():
            if alias and alias in normalized:
                broad_categories.add(broad)

        for alias, (sub, category) in self._sub_aliases.items():
            if alias and alias in normalized:
                sub_categories.add(sub)
                categories.add(category)

        return QueryContext(
            customers=customers,
            categories=categories,
            broad_categories=broad_categories,
            sub_categories=sub_categories,
        )

    def _resolve_categories(self, context: QueryContext) -> Set[str]:
        if context.categories or context.broad_categories:
            resolved: Set[str] = set(context.categories)
            for broad in context.broad_categories:
                for category, _subs in self.analytics.category_hierarchy.get(broad, {}).items():
                    resolved.add(category)
            return resolved
        return set()

    def _iter_rows(self, context: QueryContext) -> Iterable[Dict]:
        customers = context.customers
        target_categories = self._resolve_categories(context)
        sub_categories = context.sub_categories
        broad_categories = context.broad_categories
        for row in self.analytics.rows:
            if customers and row["Customer_ID"] not in customers:
                continue
            if target_categories or sub_categories or broad_categories:
                if row["Product_Category"] in target_categories:
                    yield row
                    continue
                if row["Product_Sub_Category"] in sub_categories:
                    yield row
                    continue
                if row["Broad_Category"] in broad_categories:
                    yield row
                    continue
                continue
            yield row

    def recommend_products(self, question: str, top_k: int = 5) -> Dict:
        context = self._extract_context(question)
        product_counter: Counter[str] = Counter()
        for row in self._iter_rows(context):
            product_counter[row["Order_Lines_Product_ID"]] += row["Order_Quantity"]

        if not product_counter:
            for row in self.analytics.rows:
                product_counter[row["Order_Lines_Product_ID"]] += row["Order_Quantity"]

        top_products = product_counter.most_common(top_k)
        recommendations = []
        for product_id, quantity in top_products:
            profile = self.analytics.product_profiles[product_id]
            recommendations.append(
                {
                    "product_id": product_id,
                    "product_name": profile.display_name,
                    "category": profile.category,
                    "broad_category": profile.broad_category,
                    "total_quantity": quantity,
                    "unique_customers": profile.unique_customers,
                }
            )

        customer_names = [
            self.analytics.customer_profiles[cust_id].customer_name
            for cust_id in sorted(context.customers)
        ]
        categories = sorted(self._resolve_categories(context))
        broad = sorted(context.broad_categories)

        focus_elements = []
        if customer_names:
            focus_elements.append(", ".join(customer_names))
        if categories:
            focus_elements.append(" / ".join(categories))
        elif broad:
            focus_elements.append(" / ".join(broad))

        if focus_elements:
            focus_text = " for " + " and ".join(focus_elements)
        else:
            focus_text = " across the entire dataset"

        answer = (
            f"Suggested products{focus_text}: "
            + "; ".join(
                f"{item['product_name']} ({item['category']}, qty {item['total_quantity']})"
                for item in recommendations
            )
        )

        return {
            "question": question,
            "answer": answer,
            "context": {
                "customers": sorted(context.customers),
                "categories": categories,
                "broad_categories": broad,
            },
            "recommendations": recommendations,
        }

    def multi_customer_reasoning(self, question: str, top_customers: int = 3) -> Dict:
        context = self._extract_context(question)
        customers = list(context.customers)
        if len(customers) < 2:
            extra = self._top_customers(top_customers)
            for cust_id in extra:
                if cust_id not in customers:
                    customers.append(cust_id)
                if len(customers) >= max(2, top_customers):
                    break

        categories = self._resolve_categories(context)
        if not categories:
            categories = {
                entry["product_category"]
                for entry in self.analytics.category_summary[:3]
            }

        customer_breakdown = []
        category_totals: Counter[str] = Counter()
        for cust_id in customers:
            profile = self.analytics.customer_profiles[cust_id]
            spend_by_category = {
                category: profile.category_spend.get(category, 0.0)
                for category in categories
            }
            in_scope = sum(spend_by_category.values())
            category_totals.update(spend_by_category)
            share = (in_scope / profile.total_spend) * 100 if profile.total_spend else 0.0
            customer_breakdown.append(
                {
                    "customer_id": cust_id,
                    "customer_name": profile.customer_name,
                    "total_spend": profile.total_spend,
                    "in_scope_spend": round(in_scope, 2),
                    "in_scope_share_pct": round(share, 2),
                    "category_spend": {
                        category: round(amount, 2)
                        for category, amount in spend_by_category.items()
                    },
                }
            )

        if category_totals:
            top_category, top_value = category_totals.most_common(1)[0]
        else:
            top_category, top_value = None, 0.0

        category_breakdown = [
            {
                "category": category,
                "total_spend": round(amount, 2),
                "broad_category": self._category_to_broad.get(category),
            }
            for category, amount in category_totals.items()
        ]
        category_breakdown.sort(key=lambda item: item["total_spend"], reverse=True)

        customer_names = ", ".join(
            f"{entry['customer_name']} ({entry['customer_id']})"
            for entry in customer_breakdown
        )
        category_text = ", ".join(sorted(categories))
        if top_category:
            headline = (
                f"Across {customer_names}, the focus categories {category_text} drive "
                f"${category_totals.total():,.0f} in spend. {top_category} leads with "
                f"${top_value:,.0f}."
            )
        else:
            headline = (
                f"Across {customer_names}, category spend figures were not available for the requested scope."
            )

        return {
            "question": question,
            "answer": headline,
            "context": {
                "customers": [entry["customer_id"] for entry in customer_breakdown],
                "categories": sorted(categories),
            },
            "customer_breakdown": customer_breakdown,
            "category_breakdown": category_breakdown,
        }

    def _top_customers(self, limit: int) -> List[str]:
        ordered = sorted(
            self.analytics.customer_profiles.values(),
            key=lambda profile: profile.total_spend,
            reverse=True,
        )
        return [profile.customer_id for profile in ordered[:limit]]


def _print(result: Dict, pretty: bool) -> None:
    if pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language prompt to analyse. If omitted, the script will prompt for input.",
    )
    parser.add_argument(
        "--mode",
        choices=["recommend", "compare"],
        default="recommend",
        help="Select recommendation or multi-customer reasoning mode.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of product recommendations to return.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run canned examples for both modes instead of parsing CLI input",
    )
    args = parser.parse_args(argv)
    print("Arguments:", args)

    advisor = BonusAdvisor.from_dataset()
    if args.demo:
        demo_inputs = [
            ("recommend", "Suggest high-margin add-ons for CUST_015 in the Electronics category", 5),
            ("compare", "Compare CUST_008 and CUST_014 across personal care and household essentials", args.top_k),
        ]
        for mode, prompt, top_k in demo_inputs:
            if mode == "recommend":
                result = advisor.recommend_products(prompt, top_k=top_k)
            else:
                result = advisor.multi_customer_reasoning(prompt)
            _print(result, args.pretty)
        return

    question = args.question
    if not question:
        try:
            question = input("Enter a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            print("No question provided. Exiting.")
            return

    if args.mode == "recommend":
        result = advisor.recommend_products(question, top_k=args.top_k)
    else:
        result = advisor.multi_customer_reasoning(question)
    _print(result, args.pretty)


if __name__ == "__main__":
    main()
