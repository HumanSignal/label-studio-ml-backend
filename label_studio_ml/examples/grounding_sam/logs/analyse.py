#!/usr/bin/env python3
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = 'dino_alias_log.jsonl'  # adjust if needed

# 1. Load the log records
records = []
with open(LOG_PATH, 'r') as f:
    for line in f:
        entry = json.loads(line)
        raw_phrases = entry.get('raw_phrases', [])
        mapped_labels = entry.get('mapped_labels', [])
        for raw, mapped in zip(raw_phrases, mapped_labels):
            if raw != mapped:
                records.append((raw, mapped))

# 2. Count frequency of each raw→mapped pairing
counts = Counter(records)
# Convert to DataFrame
df = pd.DataFrame(
    [(raw, mapped, cnt) for (raw, mapped), cnt in counts.items()],
    columns=['raw_phrase','mapped_label','count']
)
top20 = df.sort_values('count', ascending=False).head(20)

# 3. Print the table
print("\nTop 20 Raw → Mapped Label Mismatches:\n")
print(top20.to_string(index=False))

# 4. Save to CSV
top20.to_csv('top_mismatches.csv', index=False)
print("\n→ saved CSV: top_mismatches.csv")

# 5. Plot horizontal bar chart
plt.figure(figsize=(8,6))
labels = top20.apply(lambda r: f"{r.raw_phrase} → {r.mapped_label}", axis=1)
plt.barh(range(len(top20)), top20['count'])
plt.yticks(range(len(top20)), labels)
plt.xlabel('Count')
plt.title('Top 20 Raw→Mapped Label matches')
plt.gca().invert_yaxis()  # highest on top
plt.tight_layout()
plt.savefig('mismatch_plot.png')
print("→ saved plot: mismatch_plot.png")
plt.show()
