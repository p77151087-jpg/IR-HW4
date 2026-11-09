"""Evidence-backed interpretation and a derived report; never changes training or scoring."""
from collections import Counter
from datetime import datetime, timezone
import csv
import math
from pathlib import Path
import shutil
from uuid import uuid4
import zipfile

from .config import ROOT, RUN, ANALYSIS_WORDS, METHODS, TOPICS
from .dataset import load_data
from .evaluation import evaluate, query_bank
from .io import read_json, write_json, write_csv, sha256
from .retrieval import neighbors
from .text import document_text, terms, TOKEN_RE, normalize

ANALYSIS_ROOT = ROOT / "reports" / "analysis"
NOTES = ROOT / "resources" / "analysis_notes.json"
EVALUATION = ROOT / "reports" / "evaluation-v1"
METRIC_NAMES = {"p10": "P@10", "mrr": "MRR", "ndcg10": "nDCG@10"}
WORD_MODELS = {"cbow": "CBOW", "skipgram": "Skip-Gram"}
# The historic manifest includes UI source. Display-only edits do not change its scores.
# All other recorded sources remain checked, including every scoring/evaluation module.
DISPLAY_ONLY_SOURCES = frozenset({"ir_hw4/ui.py"})
REQUIRED_EVALUATION_SOURCES = frozenset({
    "ir_hw4/evaluation.py", "ir_hw4/retrieval.py", "ir_hw4/text.py",
    "ir_hw4/dataset.py", "ir_hw4/training.py", "ir_hw4/config.py",
    "ir_hw4/io.py", "ir_hw4/sentences.py",
})


def checked_evaluation(model_signature=None):
    manifest = read_json(EVALUATION / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Saved evaluation is incomplete")
    if manifest.get("model_manifest_sha256") != (model_signature or sha256(RUN / "manifest.json")):
        raise ValueError("Evaluation/model binding mismatch")
    if manifest.get("queries_sha256") != sha256(ROOT / "resources/queries.json"):
        raise ValueError("Evaluation/query binding mismatch")
    sources = manifest.get("source_sha256", {})
    if not REQUIRED_EVALUATION_SOURCES.issubset(sources):
        raise ValueError("Missing evaluation source fingerprints")
    for relative, digest in sources.items():
        if relative not in DISPLAY_ONLY_SOURCES and sha256(ROOT / relative) != digest:
            raise ValueError(f"Evaluation source changed: {relative}")
    for relative, digest in manifest["artifacts_sha256"].items():
        if sha256(EVALUATION / relative) != digest:
            raise ValueError(f"Saved evaluation artifact mismatch: {relative}")
    return read_json(EVALUATION / "results.json"), manifest


def analysis_signature():
    pointer = ANALYSIS_ROOT / "current.json"
    if not pointer.exists():
        return ""
    return (sha256(pointer), sha256(NOTES), sha256(Path(__file__)))


def current_analysis_directory():
    pointer = read_json(ANALYSIS_ROOT / "current.json")
    path = (ROOT / pointer["run"]).resolve()
    if not path.is_relative_to((ANALYSIS_ROOT / "runs").resolve()):
        raise ValueError("Analysis pointer escapes the reports directory")
    if sha256(path / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Analysis pointer/manifest mismatch")
    return path


def load_analysis(path=None):
    path = Path(path) if path is not None else current_analysis_directory()
    manifest = read_json(path / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Analysis report is incomplete")
    if manifest["model_manifest_sha256"] != sha256(RUN / "manifest.json"):
        raise ValueError("Analysis/model binding mismatch")
    if manifest["evaluation_manifest_sha256"] != sha256(EVALUATION / "manifest.json"):
        raise ValueError("Analysis/evaluation binding mismatch")
    if manifest["queries_sha256"] != sha256(ROOT / "resources/queries.json"):
        raise ValueError("Analysis/query binding mismatch")
    for relative, expected in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Analysis source changed: {relative}; rebuild the report")
    for relative, expected in manifest["artifacts_sha256"].items():
        if sha256(path / relative) != expected:
            raise ValueError(f"Analysis artifact mismatch: {relative}")
    checked_evaluation(manifest["model_manifest_sha256"])
    return read_json(path / "analysis.json"), path


def excerpt(document, selected_terms):
    """Return exact source substrings; split distant hits instead of inventing a sentence."""
    text = document_text(document)
    spans = []
    for word in selected_terms:
        found = next((m.span() for m in TOKEN_RE.finditer(text) if normalize(m.group()) == word), None)
        if found:
            spans.append(found)
    if not spans:
        raise ValueError("Evidence term not in source document")
    if max(end for _, end in spans) - min(start for start, _ in spans) <= 420:
        intervals = [(max(0, min(start for start, _ in spans) - 90), min(len(text), max(end for _, end in spans) + 180))]
    else:
        intervals = [(max(0, start - 70), min(len(text), end + 140)) for start, end in spans]
    return [{"id": document["id"], "title": document["title"], "start": start, "end": end,
             "text": text[start:end]} for start, end in intervals]


def pair_evidence(anchor, neighbor, documents, counts):
    together = [d for d, c in zip(documents, counts) if anchor in c and neighbor in c]
    if together:
        witnesses = excerpt(together[0], [anchor, neighbor])
    else:
        witnesses = []
        for term in (anchor, neighbor):
            document = next((d for d, c in zip(documents, counts) if term in c), None)
            if document:
                witnesses.extend(excerpt(document, [term]))
    return {
        "anchor": anchor, "neighbor": neighbor,
        "neighbor_cf": sum(c[neighbor] for c in counts),
        "neighbor_df": sum(neighbor in c for c in counts),
        "cooccurrence_documents": len(together), "witnesses": witnesses,
        "scope": "train documents; document co-occurrence, not Word2Vec-window co-occurrence",
    }


def neighbor_coordinates(rows):
    """Map center-to-neighbor distances to spokes with evenly spaced directions."""
    points = []
    for index, row in enumerate(rows):
        radius = math.sqrt(max(0., 2 * (1 - row["cosine"])))
        angle = math.pi / 2 - math.tau * index / len(rows)
        points.append({**row, "rank": index + 1, "radius": radius,
                       "x": radius * math.cos(angle), "y": radius * math.sin(angle)})
    return points


def build_payload(engine, notes, saved, documents):
    if {n["word"] for n in notes["words"]} != set(ANALYSIS_WORDS) or len(notes["words"]) != 10:
        raise ValueError("Exactly the ten fixed words need explanations")
    note_by_word = {n["word"]: n for n in notes["words"]}
    neighbors_rows = list(csv.DictReader((RUN / "word_neighbors.csv").open(encoding="utf-8-sig")))
    training = [d for d in documents if d["split"] == "train"]
    train_counts = [Counter(terms(document_text(d))) for d in training]
    words = []
    for word in ANALYSIS_WORDS:
        note = note_by_word[word]
        if any(not note.get(key, "").strip() for key in ("reasonable", "context", "unexpected")):
            raise ValueError(f"Incomplete three-part explanation: {word}")
        lists = {model: [{"rank": int(r["rank"]), "word": r["word"], "cosine": float(r["cosine"])}
                         for r in neighbors_rows if r["anchor"] == word and r["model"] == model]
                 for model in WORD_MODELS}
        if any(len(rows) != 5 for rows in lists.values()):
            raise ValueError(f"Missing Top-5: {word}")
        available = dict.fromkeys(r["word"] for rows in lists.values() for r in rows)
        if not set(note["evidence_terms"]).issubset(available):
            raise ValueError(f"Evidence term is not in either saved Top-5: {word}")
        annotations = note["annotations"]
        if annotations.keys() != available.keys() or any(
            not annotation[key].strip() for annotation in annotations.values() for key in ("relation", "reading")
        ):
            raise ValueError(f"Every saved neighbor needs a relationship and interpretation: {word}")
        evidence = [pair_evidence(word, neighbor, training, train_counts) for neighbor in available]
        words.append({**note, "neighbors": lists, "evidence": evidence})
    bank = {q["qid"]: q for q in query_bank()}
    cases = []
    for note in notes["cases"]:
        q = bank[note["qid"]]
        if any(key not in engine.by_id for key in note["document_ids"]):
            raise ValueError("Case includes a document outside the test candidate set")
        ranks = {model: {row["id"]: row for row in engine.rank(q["text"], model)[0]} for model in METHODS}
        table, evidence = [], []
        for key in note["document_ids"]:
            document = engine.by_id[key]
            shared = sorted(set(terms(q["text"])) & set(terms(document_text(document))))
            table.append({"id": key, "title": document["title"], "topics": document["topics"],
                          "topic_relevant": q["topic"] in document["topics"], "shared_query_terms": shared,
                          "ranks": {model: ranks[model][key]["rank"] if key in ranks[model] else None for model in METHODS},
                          "scores": {model: ranks[model][key]["score"] if key in ranks[model] else None for model in METHODS}})
            text = document_text(document)
            evidence.append({"id": key, "title": document["title"], "text": text[:1400],
                             "truncated": len(text) > 1400, "title_only": not bool(document["body"])})
        case_metrics = [row for row in saved["per_query"] if row["qid"] == q["qid"]]
        cases.append({**note, "query": q, "metrics": case_metrics, "documents": table, "evidence": evidence})
    # Stream original saved rankings once; preserve titles, scores and topic labels verbatim.
    top10 = []
    with (EVALUATION / "rankings.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["rank"]) <= 10:
                top10.append({**row, "rank": int(row["rank"]), "score": float(row["score"]), "relevant": int(row["relevant"])})
    return {"version": "hw4-integrated-analysis-v2", "policy": notes["policy"], "words": words, "cases": cases,
            "summary": saved["summary"], "per_query": saved["per_query"], "top10": top10}


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report_text(payload, model, dataset):
    lines = [
        "# HW4 Reuters 詞彙與文件檢索完整報告", "",
        "本實驗比較 TF-IDF、TF-IDF 加權 CBOW 與 TF-IDF 加權 Skip-Gram。詞近鄰與文件排名分開檢驗：前者說明上下文關係，後者以固定查詢與 Reuters topics 衡量排序。", "",
        "本次十查詢平均 P@10 以 Skip-Gram 較高，但 TF-IDF 的 MRR 較高；CBOW 的 nDCG@10 也略低於 TF-IDF。結果不支持所有 embedding 方法在所有指標上均改善的結論。", "",
        "## 一 資料與實驗控制", "",
        f"Reuters-21578 Distribution 1.0，ModApte：{dataset['counts']['train']:,} 篇訓練、{dataset['counts']['test']:,} 篇測試、{dataset['counts']['unused']:,} 篇 unused。",
        "按 TOPICS=YES 與 LEWISSPLIT=TRAIN/TEST 定義切分，保留空 TOPICS 元素、多標籤與其他主題的測試新聞。五主題 earn、acq、crude、grain、trade 各設兩個查詢；排名候選是完整測試集。",
        "只以訓練集建立詞表、DF、IDF 及 Word2Vec。標題與正文參與計分，日期、ID、topics 不進入向量。NFC、casefold、撇號／連字號正規化、固定 132 停用詞、不 stemming；保留 no、not、without。",
        f"訓練序列為 {model['training_sentences']:,} 句、{model['training_tokens']:,} tokens，詞表 {model['train_vocabulary']:,} 詞。句子邊界不跨文件或標題／正文；前處理後逐篇 token 與完整文字核對相同。", "",
        "## 二 方法與訓練流程", "",
        "TF 是單篇新聞或查詢的出現次數；CF 是訓練集合總次數；DF 是含詞的訓練新聞篇數。IDF(t)=log10(N_train/DF_train(t))，TF-IDF(t,d)=TF(t,d)×IDF(t)。",
        "新聞與查詢使用同一詞表、IDF 與實際重複 TF。TF-IDF cosine 是兩向量點積除以完整詞彙空間的 L2 norm。",
        "加權詞向量 v(d)=Σ[TF-IDF(t,d)×e(t)]／Σ[TF-IDF(t,d)]，求和限於有 Word2Vec 向量的詞。查詢也用相同公式，再計算原始 100 維 cosine；OOV 忽略並回報涵蓋率。",
        "CBOW 以周邊上下文預測中心詞，Skip-Gram 以中心詞預測周邊上下文。例如示意序列 crude oil prices：CBOW 可由 crude／prices 預測 oil，Skip-Gram 可由 oil 預測 crude／prices。這只是流程示意，不是新增訓練語料。",
        "兩模型均由新 Gensim Word2Vec constructor、build_vocab、train 從頭訓練；設定 negative sampling、記錄逐 epoch loss，保存完整模型並驗證重載相同。Gensim 執行訓練核心，未宣稱手寫優化器。", "",
        "| 參數 | CBOW | Skip-Gram |", "|---|---:|---:|",
    ]
    for key in ("sg", "vector_size", "window", "min_count", "epochs", "seed", "workers", "negative", "sample", "alpha", "min_alpha"):
        lines.append(f"| {key} | {model['models']['cbow']['config'][key]} | {model['models']['skipgram']['config'][key]} |")
    lines += [
        "", "其餘設定相同，包括 ns_exponent=0.75、cbow_mean=1、sorted_vocab=1、shrink_windows=True。每模型 23,172 詞；未使用預訓練或舊 HW3 權重。單 seed 與單組參數未經最佳化比較，loss 不是語意正確率。",
        "所有有效測試向量參與排序，embedding 沒有共同查詢詞門檻。零分及負 cosine 保留，零向量排除並計數；同分先按 12 位小數分數再按數字 NEWID 升冪，保存原始精度。", "",
        "## 三 固定查詢與相關性", "",
        "| 查詢 | 文字 | 目標 topic | 意圖 |", "|---|---|---|---|",
    ]
    for q in query_bank():
        lines.append(f"| {q['qid']} | {q['text']} | {q['topic']} | {q['intent']} |")
    lines += ["", "相關 iff 查詢的目標 topic 存在於原始文件 topics。這是資料集主題代理，不是針對查詢的人工判讀；分析推論不會改寫標籤或分數。", "",
              "## 四 十個詞的 Top-5 與逐詞解釋", "", payload["policy"], "",
              "表格逐項說明近鄰的關係與新聞用法。標記「意外」的列包含低頻數字、專名或容易跨主題解讀的結果。共現是兩詞同時出現的訓練新聞篇數，後面的新聞 ID 可回查原文。", "",
              "網站關係圖使用原始 100 維 cosine。各線長是中心詞與近鄰的單位向量距離 √(2 − 2 × cosine)。兩圖同尺度，連線方向由排版決定。所有近鄰的詞頻、共現與原文片段也收錄在證據包的 analysis.json。"]
    for entry in payload["words"]:
        evidence_by_word = {row["neighbor"]: row for row in entry["evidence"]}
        cbow_words = {row["word"] for row in entry["neighbors"]["cbow"]}
        shared = [row["word"] for row in entry["neighbors"]["skipgram"] if row["word"] in cbow_words]
        shared_words = "、".join(shared) if shared else "無"
        lines += ["", f"### {entry['word']}", "", f"兩模型共有 {len(shared)} 個 Top-5 鄰近詞（{shared_words}）。"]
        for name, rows in entry["neighbors"].items():
            lines += ["", f"#### {WORD_MODELS[name]}", "",
                      "| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |", "|---:|---|---:|---|---|"]
            for row in rows:
                term = row["word"]
                annotation = entry["annotations"][term]
                marker = "（意外）" if annotation.get("unexpected") else ""
                evidence = evidence_by_word[term]
                ids = "、".join(dict.fromkeys(w["id"] for w in evidence["witnesses"]))
                lines.append(f"| {row['rank']} | {md_cell(term)} | {row['cosine']:.6f} | "
                             f"**{annotation['relation']}{marker}**。{md_cell(annotation['reading'])} | "
                             f"{evidence['cooccurrence_documents']}／{ids} |")
        for key, label in (("reasonable", "合理關係"), ("context", "新聞情境"), ("unexpected", "意外結果與可能原因")):
            lines += ["", f"**{label}**　{entry[key]}"]
        examples = [evidence_by_word[term] for term in entry["evidence_terms"]]
        lines += ["", "**原文證據示例**", "", "| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |", "|---|---:|---:|---:|"]
        for evidence in examples:
            lines.append(f"| {evidence['neighbor']} | {evidence['neighbor_cf']} | {evidence['neighbor_df']} | {evidence['cooccurrence_documents']} |")
        for evidence in examples:
            lines += ["", f"證據詞：{evidence['neighbor']}"]
            for witness in evidence["witnesses"]:
                lines += ["", f"訓練新聞 {witness['id']}：{md_cell(witness['title'])}", "", "> " + witness["text"].replace("\n", " ")]
        lines += ["", "此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。"]
    lines += ["", "## 五 文件檢索效果", "",
              "P@10 分母固定為 10；MRR 對每查詢完整返回排名的 RR 取平均；nDCG@10 使用二元 gain 與 log2 折扣。平均只排除沒有 topic 正例的查詢；有正例但查詢無向量則得零分。", "",
              "| 方法 | P@10 | MRR | nDCG@10 | 查詢數 |", "|---|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        lines.append(f"| {METHODS[row['method']]} | {row['p10']:.4f} | {row['mrr']:.4f} | {row['ndcg10']:.4f} | {row['queries_included']}/{row['queries_total']} |")
    lines += ["", "### 逐查詢結果", "", "| 查詢 | 方法 | P@10 | RR | nDCG@10 |", "|---|---|---:|---:|---:|"]
    for row in payload["per_query"]:
        lines.append(f"| {row['qid']} | {METHODS[row['method']]} | {row['p10']:.4f} | {row['mrr']:.4f} | {row['ndcg10']:.4f} |")
    lines += ["", "## 六 改善與失敗案例", ""]
    for case in payload["cases"]:
        lines += [f"### {case['query']['qid']} {case['title']}", "", f"查詢：{case['query']['text']}；目標 topic：{case['query']['topic']}。", "",
                  "| 方法 | P@10 | RR | nDCG@10 |", "|---|---:|---:|---:|"]
        for row in case["metrics"]:
            lines.append(f"| {METHODS[row['method']]} | {row['p10']:.4f} | {row['mrr']:.4f} | {row['ndcg10']:.4f} |")
        lines += ["", "**已觀察到的事實**", ""]
        lines += ["- " + paragraph for paragraph in case["observations"]]
        lines += ["", "| 新聞 ID | 標題 | topics | TF-IDF 名次 | CBOW 名次 | Skip-Gram 名次 | 共同查詢詞 |",
                  "|---|---|---|---:|---:|---:|---|"]
        for row in case["documents"]:
            lines.append(f"| {row['id']} | {md_cell(row['title'])} | {', '.join(row['topics'])} | {row['ranks']['tfidf']} | {row['ranks']['cbow']} | {row['ranks']['skipgram']} | {', '.join(row['shared_query_terms'])} |")
        lines += ["", "**可能原因**　" + case["interpretation"], "", "**可推論範圍**　" + case["limitation"], ""]
        for doc in case["evidence"]:
            scope = "僅標題" if doc["title_only"] else ("標題與正文節錄" if doc["truncated"] else "標題與正文")
            lines += [f"**新聞 {doc['id']} 原文（{scope}）**", "", "> " + doc["text"].replace("\n", " "), ""]
    lines += ["## 七 限制與結論", "",
              "本次 Skip-Gram 的 P@10 與 nDCG@10 較高，但 TF-IDF 的 MRR 較高；CBOW 在 Q06 的 P@10 改善，同時仍有燃油／銅價新聞排前的局部失敗。Q09 表明相同 P@10 不等於相同排序品質。",
              "profit／loss 的近鄰與 Q01 說明：共享財務情境不等於相同損益方向。加權平均不完整保留詞序、否定、時態與複雜語意；保留否定詞本身也不代表理解了否定。",
              "topics 是粗粒度且可能多標籤的參考，不能把主題未命中直接等同自然語意錯誤。類別不平衡、每主題兩查詢的依賴、低頻詞、單 seed 與小型語料限制一般化，沒有顯著性或大型語言模型能力的推論。",
              "詞近鄰、文件共現、PCA、loss 與檢索指標分別描述不同事物。文件案例的原因解讀沒有消融實驗支持，不宣稱單一近鄰或某個詞已被證明造成改善。", "",
              "## 八 重現與來源", "",
              "程式、獨立虛擬環境與資料位於 IR-HW4，未依賴 HW3 執行環境。原始模型與 evaluation-v1 實驗保留；本報告只從它們衍生，完整綁定與檔案 SHA256 見同目錄 manifest.json。", "",
              "執行 scripts/test_hw4.ps1 可測試四頁及計算；scripts/verify_hw4.py 重算數值。scripts/build_report.py 另建新的報告版本，scripts/verify_analysis.py 核對詞頻、原文片段、案例名次及整合報告。網站只讀完成的報告，不在 rerun 重新建檔。",
              f"語料 SHA256：{dataset['documents_sha256']}",
              f"模型 manifest SHA256：{sha256(RUN / 'manifest.json')}",
              f"原評估 manifest SHA256：{sha256(EVALUATION / 'manifest.json')}", "",
              "[Reuters-21578 官方原始 README](https://kdd.ics.uci.edu/databases/reuters21578/README.txt)",
              "[Gensim Word2Vec 訓練 API](https://radimrehurek.com/gensim/models/word2vec.html)", "",
              "## 附錄 三方法對十查詢的完整 Top 10", "",
              "| 查詢 | 方法 | 名次 | 新聞 ID | cosine | 目標 topic 命中 | 標題 |", "|---|---|---:|---|---:|---:|---|"]
    for row in payload["top10"]:
        lines.append(f"| {row['qid']} | {row['method']} | {row['rank']} | {row['id']} | {row['score']:.6f} | {row['relevant']} | {md_cell(row['title'])} |")
    return "\n".join(lines) + "\n"


def build_report(output=None):
    from .training import load_engine
    engine, model = load_engine()
    saved, _ = checked_evaluation()
    rows, summary, _ = evaluate(engine)
    if rows != saved["per_query"] or summary != saved["summary"]:
        raise ValueError("Current retrieval differs from saved experiment")
    documents, dataset = load_data()
    notes = read_json(NOTES)
    if notes["model_manifest_sha256"] != sha256(RUN / "manifest.json") or notes["evaluation_manifest_sha256"] != sha256(EVALUATION / "manifest.json"):
        raise ValueError("Interpretation notes were written for a different experiment")
    payload = build_payload(engine, notes, saved, documents)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    output = Path(output or ANALYSIS_ROOT / "runs" / run_name).resolve()
    if not output.is_relative_to((ANALYSIS_ROOT / "runs").resolve()):
        raise ValueError("Derived report must stay inside reports/analysis/runs")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "analysis.json", payload)
    (output / "HW4_REPORT.md").write_text(report_text(payload, model, dataset), encoding="utf-8")
    for name in ("metrics_per_query.csv", "metrics_summary.csv", "queries.json"):
        shutil.copyfile(EVALUATION / name, output / name)
    shutil.copyfile(RUN / "word_neighbors.csv", output / "word_neighbors.csv")
    sources = {"ir_hw4/analysis.py": sha256(Path(__file__)), "resources/analysis_notes.json": sha256(NOTES)}
    binding = {"model_manifest_sha256": sha256(RUN / "manifest.json"),
               "evaluation_manifest_sha256": sha256(EVALUATION / "manifest.json"),
               "queries_sha256": sha256(ROOT / "resources/queries.json"),
               "corpus_sha256": model["corpus_sha256"], "source_sha256": sources}
    write_json(output / "sources.json", binding)
    with zipfile.ZipFile(output / "HW4_REPORT.zip", "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                package.write(path, path.name)
    manifest = {"version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **binding, "word_count": len(payload["words"]), "case_qids": [c["query"]["qid"] for c in payload["cases"]],
                "interpretation_policy": payload["policy"],
                "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    load_analysis(output)
    temporary = ANALYSIS_ROOT / "current.tmp.json"
    write_json(temporary, {"run": output.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(output / "manifest.json")})
    temporary.replace(ANALYSIS_ROOT / "current.json")
    return output


def verify_analysis(path=None):
    """Recalculate evidence from the corpus and ranks, separately from report assembly."""
    from .training import load_engine
    payload, directory = load_analysis(path)
    engine, _ = load_engine()
    documents, _ = load_data()
    by_id = {d["id"]: d for d in documents}
    training = [d for d in documents if d["split"] == "train"]
    train_counts = [Counter(terms(document_text(d))) for d in training]
    witness_count, pair_count, score_count, neighbor_count = 0, 0, 0, 0
    saved_neighbors = list(csv.DictReader((RUN / "word_neighbors.csv").open(encoding="utf-8-sig")))
    assert [entry["word"] for entry in payload["words"]] == ANALYSIS_WORDS
    for entry in payload["words"]:
        available = {row["word"] for rows in entry["neighbors"].values() for row in rows}
        assert set(entry["annotations"]) == available
        assert {row["neighbor"] for row in entry["evidence"]} == available
        for name, rows in entry["neighbors"].items():
            expected = [r for r in saved_neighbors if r["anchor"] == entry["word"] and r["model"] == name]
            computed = neighbors(*engine.models[name], entry["word"])
            assert [r["rank"] for r in rows] == list(range(1, 6))
            assert [r["word"] for r in rows] == [r["word"] for r in expected]
            assert [r["word"] for r in rows] == [r["word"] for r in computed]
            assert all(abs(row["cosine"] - float(ref["cosine"])) < 1e-14 for row, ref in zip(rows, expected))
            assert all(abs(row["cosine"] - ref["cosine"]) < 1e-12 for row, ref in zip(rows, computed))
            neighbor_count += len(rows)
        for evidence in entry["evidence"]:
            word, neighbor = entry["word"], evidence["neighbor"]
            assert evidence["neighbor_cf"] == sum(c[neighbor] for c in train_counts)
            assert evidence["neighbor_df"] == sum(neighbor in c for c in train_counts)
            assert evidence["cooccurrence_documents"] == sum(word in c and neighbor in c for c in train_counts)
            pair_count += 1
            witness_terms = set()
            for witness in evidence["witnesses"]:
                doc = by_id[witness["id"]]
                assert doc["split"] == "train"
                assert document_text(doc)[witness["start"]:witness["end"]] == witness["text"]
                witness_terms.update(terms(witness["text"]))
                witness_count += 1
            assert {word, neighbor}.issubset(witness_terms)
    results, _ = checked_evaluation()
    for case in payload["cases"]:
        query = case["query"]
        assert query == next(q for q in query_bank() if q["qid"] == query["qid"])
        assert case["metrics"] == [r for r in results["per_query"] if r["qid"] == query["qid"]]
        ranked = {name: {r["id"]: r for r in engine.rank(query["text"], name)[0]} for name in METHODS}
        for row in case["documents"]:
            assert row["topics"] == by_id[row["id"]]["topics"]
            assert row["topic_relevant"] == (query["topic"] in row["topics"])
            for method in METHODS:
                assert row["ranks"][method] == ranked[method][row["id"]]["rank"]
                assert abs(row["scores"][method] - ranked[method][row["id"]]["score"]) < 1e-12
                score_count += 1
        for witness in case["evidence"]:
            doc = by_id[witness["id"]]
            assert doc["split"] == "test" and witness["text"] == document_text(doc)[:1400]
            assert witness["title_only"] == (not bool(doc["body"]))
    with zipfile.ZipFile(directory / "HW4_REPORT.zip") as package:
        expected_files = {p.name for p in directory.iterdir() if p.is_file() and p.name not in {"manifest.json", "HW4_REPORT.zip"}}
        assert set(package.namelist()) == expected_files
        for name in package.namelist():
            assert package.read(name) == (directory / name).read_bytes()
    return {"status": "passed", "report": directory.relative_to(ROOT).as_posix(),
            "words_explained": len(payload["words"]), "neighbor_rows": neighbor_count,
            "word_pairs_verified": pair_count, "source_excerpts_verified": witness_count,
            "case_count": len(payload["cases"]), "case_scores_verified": score_count,
            "zip_matches_report_and_data": True, "report_sha256": sha256(directory / "HW4_REPORT.md")}
