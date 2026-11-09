"""Topic-based retrieval evaluation; these labels are not human query judgments."""
from datetime import datetime, timezone
import math
from pathlib import Path
import numpy as np

from .config import ROOT, METHODS, TOPICS
from .io import read_json, sha256, write_json, write_csv


def query_bank():
    return read_json(ROOT / "resources" / "queries.json")["queries"]


def metrics(ranked_ids, relevant_ids, k=10):
    if k < 1 or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Invalid k or duplicate ranked IDs")
    relevant = set(relevant_ids)
    if not relevant:
        return {"p10": None, "mrr": None, "ndcg10": None, "relevant_total": 0}
    hits = [int(doc_id in relevant) for doc_id in ranked_ids]
    rr = next((1. / rank for rank, hit in enumerate(hits, 1) if hit), 0.)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:k], 1))
    ideal = sum(1. / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {"p10": sum(hits[:k]) / k, "mrr": rr, "ndcg10": dcg / ideal, "relevant_total": len(relevant)}


def evaluate(engine):
    rows, ranking_rows = [], []
    for query in query_bank():
        relevant = {d["id"] for d in engine.documents if query["topic"] in d["topics"]}
        for method in METHODS:
            ranked, status = engine.rank(query["text"], method)
            result = metrics([r["id"] for r in ranked], relevant)
            rows.append({**query, "method": method, **result, "valid_query": status["valid_query"],
                         "known_tokens": status["known_tokens"], "total_tokens": status["total_tokens"],
                         "embedding_weight_coverage": status["embedding_weight_coverage"],
                         "retrieved": len(ranked), "valid_documents": status["valid_documents"],
                         "included": bool(relevant)})
            for row in ranked:
                ranking_rows.append({"qid": query["qid"], "query": query["text"], "topic": query["topic"],
                                     "method": method, **row, "relevant": int(row["id"] in relevant)})
    summary = []
    for method in METHODS:
        included = [r for r in rows if r["method"] == method and r["included"]]
        summary.append({"method": method, "queries_total": len(query_bank()), "queries_included": len(included),
                        **{key: float(np.mean([r[key] for r in included])) if included else None
                           for key in ("p10", "mrr", "ndcg10")}})
    return rows, summary, ranking_rows


def save_evaluation(engine, run, output=None):
    output = Path(output or ROOT / "reports" / "evaluation-v1")
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite saved experiment: {output}")
    output.mkdir(parents=True)
    rows, summary, rankings = evaluate(engine)
    write_csv(output / "metrics_per_query.csv", rows)
    write_csv(output / "metrics_summary.csv", summary)
    write_csv(output / "rankings.csv", rankings)
    write_json(output / "results.json", {"per_query": rows, "summary": summary})
    write_json(output / "queries.json", read_json(ROOT / "resources" / "queries.json"))
    topic_counts = {topic: sum(topic in d["topics"] for d in engine.documents) for topic in TOPICS}
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "ir_hw4").glob("*.py"))}
    manifest = {"status": "complete", "source": "reuters_topics", "k": 10,
                "queries_sha256": sha256(ROOT / "resources" / "queries.json"),
                "model_manifest_sha256": sha256(Path(run) / "manifest.json"),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_documents": len(engine.documents), "topic_counts": topic_counts,
                "methods": METHODS, "source_sha256": sources,
                "policy": "P@10 denominator is always 10; MRR uses full returned ranking; binary nDCG@10 uses log2; macro average over queries with >=1 known topic positive. OOV queries score zero, not excluded.",
                "limitation": "Topics are dataset-provided coarse, multilabel proxies, not manual relevance judgments for the exact query; a matching topic does not guarantee the query's specific intent.",
                "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    write_report(engine, summary, rows, rankings, output)
    manifest["artifacts_sha256"]["REPORT.md"] = sha256(output / "REPORT.md")
    write_json(output / "manifest.json", manifest)
    return summary


def write_report(engine, summary, rows, rankings, output):
    text = ["# HW4 Reuters 新聞檢索實驗", "", "以 ModApte 訓練集建立詞表、IDF 及兩組新 Word2Vec；對完整 3,299 篇測試新聞排序。模型未使用測試集訓練，topics 不進入文件或查詢向量。", "",
            "TF-IDF=f×log10(N_train/DF_train)。文件與查詢同樣切詞、移除固定停用詞、不 stemming。加權詞向量以 TF-IDF 加權平均有向量的詞，再計算原維度 cosine。", "",
            "## 文件檢索結果", "", "| 方法 | P@10 | MRR | nDCG@10 | 查詢數 |", "|---|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f'| {METHODS[r["method"]]} | {r["p10"]:.4f} | {r["mrr"]:.4f} | {r["ndcg10"]:.4f} | {r["queries_included"]}/{r["queries_total"]} |')
    baseline = summary[0]
    text += ["", "## 結果解讀", ""]
    for result in summary[1:]:
        differences = "、".join(f'{name} {result[key]-baseline[key]:+.4f}' for key, name in
                                 (("p10", "P@10"), ("mrr", "MRR"), ("ndcg10", "nDCG@10")))
        text.append(f'- {METHODS[result["method"]]} 相對 TF-IDF：{differences}。')
    text += ["", "這些是固定十個查詢、單一 seed、單組超參數下的觀察，沒有顯著性或架構普遍優劣的結論。加權平均無法保留詞序、否定與複雜語意，embedding 不保證改善排名。", "",
             "## 實際新聞案例", ""]
    for query in query_bank():
        text.append(f'### {query["qid"]} {query["text"]}（{query["topic"]}）')
        for method in METHODS:
            top = [r for r in rankings if r["qid"] == query["qid"] and r["method"] == method][:3]
            text.append(f'- {METHODS[method]}：' + "；".join(f'{r["id"]} {r["title"]}（cos={r["score"]:.4f}、topic相關={r["relevant"]}）' for r in top))
        text.append("")
    text += ["## 詞彙層級與限制", "",
             "詞 Top-5 近鄰另存 word_neighbors.csv；請以完整新聞上下文判斷同義、反義、共現或相似情境。profit/loss 的接近可能來自相似財務句式，不能推論兩者可交換或檢索同樣的新聞。", "",
             "相關性使用資料集 topics 的二元主題代理，未宣稱逐查詢人工判讀。跨主題新聞可有多標籤；候選集合保留其他主題及空 topics 的測試文件，避免只在正例中搜尋。類別不平衡及同主題兩查詢間的依賴會限制平均值解讀。", "",
             "保留 OOV、零向量與 TF-IDF 詞彙權重涵蓋紀錄；有效 embedding 文件的負 cosine 也參與排序。所有結果均來自原維度向量，PCA 只供展示。", "",
             "資料來源：https://kdd.ics.uci.edu/databases/reuters21578/README.txt",
             "Word2Vec：https://radimrehurek.com/gensim/models/word2vec.html", ""]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
