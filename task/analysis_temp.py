import pandas as pd
import json
from itertools import combinations
from collections import Counter
import numpy as np
from pathlib import Path

DATA_PATH = Path('Dataset_product_orders.csv')

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
for col in ['order_quantity','subtotal','total_orderquantity','total_subtotal','order_lines_unit_price']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

customer_spend = df.groupby('customer_id')['total_subtotal'].sum()
threshold = customer_spend.quantile(0.9)
high_value_ids = customer_spend[customer_spend >= threshold].index
hv = df[df['customer_id'].isin(high_value_ids)].copy()
hv['order_key'] = hv['customer_id'].astype(str) + '_' + hv['date'].dt.strftime('%Y-%m-%d')

order_products = hv.groupby('order_key')['product_display_name'].agg(lambda s: sorted(set(s.dropna())))
pair_counter = Counter()
for products in order_products:
    if len(products) < 2:
        continue
    for a, b in combinations(products, 2):
        pair_counter[(a, b)] += 1

top_pairs = [
    {'product_a': a, 'product_b': b, 'orders': count}
    for (a, b), count in pair_counter.most_common(15)
]

max_date = df['date'].max()
cutoff = max_date - pd.DateOffset(months=3)
recent = df[df['date'] >= cutoff].copy()
recent['order_quantity'] = recent['order_quantity'].astype(float)
recent['subtotal'] = recent['subtotal'].astype(float)
recent['order_quantity_z'] = (recent['order_quantity'] - recent['order_quantity'].mean()) / recent['order_quantity'].std(ddof=0)
recent['subtotal_z'] = (recent['subtotal'] - recent['subtotal'].mean()) / recent['subtotal'].std(ddof=0)
recent['composite_z'] = recent[['order_quantity_z', 'subtotal_z']].abs().max(axis=1)
outliers = recent[recent['composite_z'] > 2].sort_values('composite_z', ascending=False)
outlier_rows = outliers.head(15)[['date','customer_id','customer','product_display_name','order_quantity','subtotal','composite_z']]

outlier_records = []
for row in outlier_rows.to_dict(orient='records'):
    serialised = {}
    for key, value in row.items():
        if hasattr(value, 'isoformat'):
            serialised[key] = value.isoformat()
        elif isinstance(value, (np.floating, np.integer)):
            serialised[key] = float(value)
        else:
            serialised[key] = value
    outlier_records.append(serialised)

payload = {
    'metadata': {
        'high_value_threshold': float(threshold),
        'recent_window_start': cutoff.date().isoformat(),
        'records_indexed': int(df.shape[0])
    },
    'high_value_pairs': top_pairs,
    'recent_outliers': outlier_records
}

print(json.dumps(payload, indent=2))
