import pandas as pd

# load JSONL
df = pd.read_json('dino_alias_log.jsonl', lines=True)

# explode so each phrase ↔ label is its own row
exploded = (
    df
    .explode('raw_phrases')
    .explode('mapped_labels')
    .loc[:, ['raw_phrases','mapped_labels']]
)

# frequency of raw phrases
phrase_counts = (
    exploded
    .groupby('raw_phrases')
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
)
print("Top raw phrases:")
print(phrase_counts.head(10))

# frequency of mappings
mapping_counts = (
    exploded
    .groupby(['raw_phrases','mapped_labels'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
)
print("\nTop raw→mapped pairs:")
print(mapping_counts.head(10))

# Save to CSV for easier browsing
phrase_counts.to_csv('phrase_counts.csv', index=False)
mapping_counts.to_csv('mapping_counts.csv', index=False)
