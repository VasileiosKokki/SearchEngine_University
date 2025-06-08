def evaluate_topk_results(golden_answers, topk_results):
    total_precision = 0
    num_queries = len(golden_answers)

    for qid in golden_answers:
        correct_set = set(golden_answers[qid])  # relevant docs
        _, ranked_docs = topk_results[qid]
        retrieved_set = set(doc_id for doc_id, _ in ranked_docs)        # top-k retrieved docs

        # Compute TP, FP, FN
        tp = len(correct_set & retrieved_set)
        fp = len(retrieved_set - correct_set)

        # Compute Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        total_precision += precision

    # Compute means
    mean_precision = total_precision / num_queries
    return mean_precision