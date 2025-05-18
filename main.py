import torch

from functions import *

def startup():

    # ---------- 1) Define representations of the texts ----------

    stopwords = load_stopwords()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=device)
    docs_metadata = load_all_docs_metadata('wiki_movie_plots_deduped.csv')


    if not os.path.exists("corpus_index.pkl") or not os.path.exists("idf_values.pkl"):
        save_bm25_representation(stopwords, docs_metadata)

    if not os.path.exists("sentence_embeddings.pkl"):
        save_sentence_embeddings(model, docs_metadata)


    with open("corpus_index.pkl", "rb") as file:
        corpus_index = pickle.load(file)

    with open("idf_values.pkl", "rb") as file:
        idf_values = pickle.load(file)


    with open("sentence_embeddings.pkl", "rb") as file:
        sentence_embeddings = pickle.load(file)

    # ---------- Queries ----------
    # # ACTION: open the queries file and read three queries
    queries = {}
    with open('dev.titles.queries', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            q_id = f"Q{i+1}"
            q_text = line.strip()
            queries[q_id] = q_text


    # ---------- 3) Implement the mechanism for retrieving and calibrating the results ----------

    doc_ids = list(docs_metadata.keys())

    avgdl = compute_avgdl(corpus_index)

    top10_results_MiniLM = get_top10_results_per_query(
                                queries=queries,
                                doc_ids=doc_ids,
                                sentence_embeddings=sentence_embeddings,
                                model=model,
                                model_name=model_name
                            )

    top10_results_BM25 = get_top10_results_per_query(
                                queries=queries,
                                doc_ids=doc_ids,
                                model_name="bm25",
                                corpus_index=corpus_index,
                                stopwords=stopwords,
                                idf_values=idf_values,
                                avgdl=avgdl
                            )

    path1 = f'dev.3-2-1_bm25.qrel'
    path2 = f'dev.3-2-1_{model_name}.qrel'
    if not os.path.exists("golden_answers"):
        merge_and_filter_golden_answers(path1, path2)

    golden_answers = {}
    # we assume that all the retrieved are somewhat relevant
    with open("golden_answers", 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split()
            # Assuming the ID is the first field, title is third and abstract is the fourth field
            q_id = fields[0].strip()
            doc_id = fields[1].strip()
            score = int(fields[2].strip())
            if q_id in golden_answers.keys():
                golden_answers[q_id].append((doc_id, score))
            else:
                golden_answers[q_id]=[]
                golden_answers[q_id].append((doc_id, score))



    evaluate_topk_results(golden_answers, top10_results_MiniLM, model_name=model_name)
    evaluate_topk_results(golden_answers, top10_results_BM25, model_name="bm25")





    return corpus_index, idf_values, doc_ids, stopwords, avgdl, sentence_embeddings, model, docs_metadata