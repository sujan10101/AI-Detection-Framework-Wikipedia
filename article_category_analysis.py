"""
Category Analysis: Which Wikipedia categories are most AI-flagged?
Filters is_ai_flagged == 1, explodes pipe-separated categories,
and ranks them by frequency — excluding Wikipedia maintenance/metadata categories.
"""

import pandas as pd

DATASET_PATH = "dataset_balanced.csv"
TOP_N = 50

df = pd.read_csv(DATASET_PATH)

print(f"Total articles      : {len(df)}")
print(f"AI-flagged (1)      : {df['is_ai_flagged'].sum()}")
print(f"Human-written (0)   : {(df['is_ai_flagged'] == 0).sum()}")
print()

# Keep only AI-flagged rows
ai_df = df[df['is_ai_flagged'] == 1].copy()
ai_df = ai_df.dropna(subset=['categories'])

# Split pipe-separated categories and explode
ai_df['category'] = ai_df['categories'].str.split(' | ', regex=False)
exploded = ai_df.explode('category')
exploded['category'] = exploded['category'].str.strip()
exploded = exploded[exploded['category'] != '']

# Count AI-flagged occurrences per category
category_counts = (
    exploded['category']
    .value_counts()
    .reset_index()
)
category_counts.columns = ['category', 'ai_flagged_count']

# Compute total articles per category (across all labels)
all_df = df.dropna(subset=['categories']).copy()
all_df['category'] = all_df['categories'].str.split(' | ', regex=False)
all_exploded = all_df.explode('category')
all_exploded['category'] = all_exploded['category'].str.strip()
all_exploded = all_exploded[all_exploded['category'] != '']

total_per_cat = all_exploded.groupby('category').size().reset_index(name='total_count')
category_counts = category_counts.merge(total_per_cat, on='category', how='left')
category_counts['ai_flagged_pct'] = (
    category_counts['ai_flagged_count'] / category_counts['total_count'] * 100
).round(1)

# ── Filter out Wikipedia maintenance/metadata categories ────────────────────
EXCLUDE_PREFIXES = [
    "articles ", "all articles", "pages ", "use ", "cs1 ", "short description",
    "wikipedia ", "webarchive", "template ", "infobox", "wikidata",
    "good articles", "featured articles", "orphaned articles",
    "articles with ", "all wikipedia", "articles containing",
    "articles lacking", "articles needing", "articles to be",
    "articles that ", "articles which", "redirects ", "coordinates ",
    "wikipedia articles", "official website", "vcard", "hcard",
    "use mdy", "use dmy", "use british", "use american",
    "blp ", "blp articles", "all blp",
]

def is_metadata(cat: str) -> bool:
    c = cat.lower()
    return any(c.startswith(p) for p in EXCLUDE_PREFIXES)

content_counts = category_counts[~category_counts['category'].apply(is_metadata)].copy()
content_counts = content_counts.reset_index(drop=True)

print(f"Unique categories (total)                      : {len(category_counts)}")
print(f"After removing metadata/maintenance categories : {len(content_counts)}")
print()

# ── Top N content categories by AI-flagged count ────────────────────────────
print("=" * 80)
print(f"TOP {TOP_N} CONTENT CATEGORIES BY AI-FLAGGED ARTICLE COUNT")
print("=" * 80)
print(f"{'Rank':<5} {'AI Count':>8} {'Total':>7} {'AI %':>6}  Category")
print("-" * 80)
for rank, (_, row) in enumerate(content_counts.head(TOP_N).iterrows(), 1):
    print(f"{rank:<5} {int(row['ai_flagged_count']):>8} {int(row['total_count']):>7} {row['ai_flagged_pct']:>5.1f}%  {row['category']}")

# ── Top 20 content categories where >80% are AI-flagged (min 10 articles) ───
print()
print("=" * 80)
print("TOP 20 CONTENT CATEGORIES WHERE >80% OF ARTICLES ARE AI-FLAGGED (min 10)")
print("=" * 80)
high_ai = content_counts[
    (content_counts['ai_flagged_pct'] >= 80) &
    (content_counts['ai_flagged_count'] >= 10)
].sort_values(['ai_flagged_pct', 'ai_flagged_count'], ascending=[False, False]).head(20)

if high_ai.empty:
    print("None found.")
else:
    print(f"{'Rank':<5} {'AI Count':>8} {'Total':>7} {'AI %':>6}  Category")
    print("-" * 80)
    for rank, (_, row) in enumerate(high_ai.iterrows(), 1):
        print(f"{rank:<5} {int(row['ai_flagged_count']):>8} {int(row['total_count']):>7} {row['ai_flagged_pct']:>5.1f}%  {row['category']}")

# ── Save both results ────────────────────────────────────────────────────────
category_counts.to_csv("category_ai_analysis.csv", index=False)
content_counts.to_csv("category_ai_analysis_content_only.csv", index=False)
print()
print("Full results saved to       : category_ai_analysis.csv")
print("Content-only results saved  : category_ai_analysis_content_only.csv")
