""" This file contains the prompts for OpenAI to label RAG texts for positive, hard negative, or neutral relevance.
"""


def classification_prompt(query, texts):
    return (
        "1. Assess the accuracy and relevance of the TEXTS that were retrieved accordingly to the QUERY.\n"
        "2. For each TEXT you should output a relevance score as the `score` field between 0.0 and 1.0, "
        "where 0.0 is the worse, 1.0 is the best.\n"
        "3. For each TEXT you should define a label: \n"
        "  - If the TEXT contains information that can be used to provide a clear answer to the QUERY, "
        "it's the `positives` label,\n"
        "  - If the TEXT doesn't contain information that would answer the QUERY, "
        "it's the `hard_negatives` label,\n"
        "  - otherwise it's the `neutral` label,\n"
        "output the label as the `label` field.\n"
        "5. Each TEXT starts with `===>`, a number and a newline, "
        "then a text fragment follows (e.g. `\n===> 1.\n this is text one\n`), "
        "you should use this number in the JSON output as the `id` field.\n"
        "6. The output must be in JSON format only, never use ``` in the beginning of output, write JSON as is, "
        "e.g.:\n\n"
        "[\n"
        '{"id": 1, "score": 0.9, "label": "positives"},\n'
        '{"id": 2, "score": 0.1, "label": "hard_negatives"},\n'
        '{"id": 3, "score": 0.5, "label": "neutral"},\n'
        "]\n\n"
        "---------------------------\n"
        # you can add your task context here to improve the result quality
        # -----------------------------
        "QUERY:\n"
        f"{query}\n"
        "---------------------------\n"
        "TEXTS:\n"
        + ("".join([f"\n\n===> {i}.\n{text}" for i, text in enumerate(texts)]))
    )
