import re

import requests


def judge_single_query(q_text, docs_with_meta, api_key):
    doc_ids = [doc['doc_id'] for doc in docs_with_meta]
    judgment_text = get_judgements_from_llm(q_text, docs_with_meta, api_key)
    relevances = parse_llm_response_to_list(judgment_text, expected_count=len(docs_with_meta))

    docs = [
        {"doc_id": doc_id, "relevant": relevance}
        for doc_id, relevance in zip(doc_ids, relevances)
    ]

    return docs


def merge_queries_top10_judged(*judged_sources):
    merged = {}

    for source in judged_sources:
        for q_id, (query_text, docs) in source.items():
            if q_id not in merged:
                merged[q_id] = {"query": query_text, "docs": []}
            else:
                # Optionally update the query text (prioritize earlier sources)
                if not merged[q_id]["query"]:
                    merged[q_id]["query"] = query_text

            seen_doc_ids = {doc["doc_id"] for doc in merged[q_id]["docs"]}

            for doc in docs:
                if doc["doc_id"] not in seen_doc_ids:
                    merged[q_id]["docs"].append(doc)
                    seen_doc_ids.add(doc["doc_id"])

    return merged


def save_golden_answers(queries_top10_judged):
    with open("golden_answers.txt", "a") as f:
        for qid, data in queries_top10_judged.items():
            for doc in data["docs"]:
                if doc["relevant"].lower() == "yes":
                    f.write(f"{qid} {doc['doc_id']}\n")


def retrieve_golden_answers():
    golden_answers = {}
    with open("golden_answers.txt", 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue  # skip malformed lines
            q_id = fields[0].strip()
            doc_id = fields[1].strip()
            if q_id not in golden_answers:
                golden_answers[q_id] = []
            golden_answers[q_id].append(doc_id)
    return golden_answers


def parse_llm_response_to_list(response_text, expected_count=10):
    results = ["no"] * expected_count
    lines = response_text.strip().lower().splitlines()

    for line in lines:
        line = line.strip()  # ← this is critical
        match = re.match(r"document\s+(\d+)\s*:\s*(yes|no)", line)
        if match:
            idx = int(match.group(1)) - 1
            label = match.group(2)
            if 0 <= idx < expected_count:
                results[idx] = label

    return results


def get_judgements_from_llm(query, docs, api_key):
    doc_list = "\n".join(
        f"Document {i+1}:\n{doc['evaluation']}\n---" for i, doc in enumerate(docs)
    )

    prompt = f"""You are an expert in document retrieval evaluation.

Given the query:
"{query}"

And the following 10 documents:

{doc_list}

For each document, respond with "yes" or "no" to indicate whether it is relevant to the query. Respond in the format:

Document 1: yes
Document 2: no
...
Document 10: yes
"""

    # Set the Hugging Face API endpoint and your API key
    model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "inputs": prompt,  # Adding 'inputs' for the model to understand the input
        "parameters": {
            "max_length": 512,
        }
    }
    # Send the request to Hugging Face API
    response = requests.post(model_url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        return response_data[0]["generated_text"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None