"""Reuters analysis and retrieval, with explicit submission of new experiments."""
from collections import Counter
from dataclasses import replace
from html import escape
import io
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from .config import ROOT, RUN, METHODS, TOPICS, ANALYSIS_WORDS, README_URL
from .dataset import load_data
from .embedding_weighting import COMPARISON, read_comparison, search_query_bank, unweighted_scores, weighted_query_score
from .evaluation import query_bank, evaluate, metrics
from .experiment_ui import (
    DEFAULT_MODEL_RUN, REFERENCE_RUNS, checked_bytes, compare_extreme_losses, completed_manifests,
    experiment_panel, model_setting_description, model_setting_label, reference_model_extremes, website_experiment_configs,
)
from .experiments import read_experiment
from .io import read_json, sha256
from .retrieval import neighbors, unit_rows
from .sentences import split_sentences
from .text import TOKEN_RE, terms, document_blocks, document_text, normalize
from .training import load_engine
from .analysis import WORD_MODELS, analysis_signature, load_analysis, checked_evaluation, excerpt, pair_evidence
from .word_projection import get_vocabulary_projection, project_neighbor_space

PAGES = ["資料概況", "詞向量分析", "新聞搜尋與比較", "效果評估"]
COLORS = ["#173F50", "#C66B3D", "#749385"]
METRICS = {"p10": "Precision@10", "mrr": "MRR", "ndcg10": "nDCG@10"}
SEARCH_METHOD_NAMES = {"tfidf": "TF-IDF", **{key: f"加權 {label}" for key, label in WORD_MODELS.items()}}
SEARCH_QUERY_TYPES = {"keyword": "關鍵字", "fuzzy": "模糊句", "mixed": "混用"}
WORD_QUESTIONS = {
    "reasonable": "哪些鄰近詞具有合理的語意或使用情境關係？",
    "context": "哪些結果可能來自經常出現的相似新聞情境？",
    "unexpected": "哪些結果不符合預期？可能原因為何？",
}
CORPUS_SOURCE = "周邊用詞統計"
CONTEXT_RADIUS = 2
# Context examples from the official per-neighbor readings. A term can also
# answer question 1 or 3. Counts are displayed separately from this reading.
CONTEXT_NEIGHBORS = {
    "profit": {"loss", "vs", "net"},
    "loss": {"oper", "vs", "profit", "net"},
    "earnings": {"revenues", "results", "losses", "full-year"},
    "acquisition": {"merger", "transaction", "purchase", "buyout"},
    "merger": {"acquisition", "buyout", "kistler", "str", "rni", "vons", "rainbow's"},
    "oil": {"natural", "gas", "shut-in", "liquids", "kernel", "inter-city"},
    "petroleum": {"gas", "esso", "columbia", "wpla", "clnp", "norte"},
    "wheat": {"barley", "corn", "soybeans", "winter"},
    "grain": {"soybeans", "sugar"},
    "trade": {"invisibles", "retaliation", "gatt", "warn", "full-scale", "aggravation", "tensions", "deplore"},
}
TSNE_TOPIC_WORDS = {
    "穀物與農作": ("wheat", "corn", "barley", "oats", "sorghum", "soybeans", "maize", "rice", "grain", "rapeseed"),
    "航運": ("ship", "ships", "shipping", "vessels", "vessel", "tanker", "tankers"),
    "貨幣與匯率": ("dollar", "yen", "currency", "currencies", "sterling", "franc", "exchange", "devaluation"),
    "貴金屬報價": ("gold", "silver", "platinum", "palladium", "bullion", "ounce", "ounces"),
    "原油與天然氣": ("oil", "crude", "petroleum", "gas", "propane", "liquids", "shut-in", "kerosene", "barrels"),
    "企業財報": ("earnings", "profit", "profits", "income", "loss", "losses", "revenues", "sales", "operating"),
    "企業併購": ("merger", "acquisition", "buyout", "takeover", "takeovers", "acquisitions", "mergers", "bid", "tender"),
}


@st.cache_resource(show_spinner="正在載入 Reuters 與兩組詞向量…")
def resources(signature):
    engine, manifest = load_engine()
    documents, data_manifest = load_data()
    return engine, manifest, documents, data_manifest


@st.cache_resource(show_spinner="正在載入已訓練的模型…", max_entries=3)
def experiment_resources(run, signature):
    return load_engine(run)


def checked_experiment_resources(name):
    # Recheck the files even when the in-memory model is already cached.
    run, _, _, result = read_experiment(name)
    engine, manifest = experiment_resources(run, sha256(run / "manifest.json"))
    return engine, manifest, result, run


def distinct_run_labels(labels):
    counts = Counter(labels.values())
    return {name: f"{label}（{name}）" if counts[label] > 1 else label for name, label in labels.items()}


def model_run_selectbox(col, method, configs, marks, key):
    label = WORD_MODELS[method]
    if not configs:
        col.info(f"目前沒有可選的 {label} 實驗。")
        return None
    labels = distinct_run_labels({name: model_setting_description(config) for name, config in configs.items()})
    picker_labels = {name: f"{text} ({marks[name, method]})" if (name, method) in marks else text
                     for name, text in labels.items()}
    names = list(configs)
    default = next((name for (name, model), mark in marks.items()
                    if model == method and mark.endswith("最高")), DEFAULT_MODEL_RUN)
    return col.selectbox(label, names, index=names.index(default) if default in names else 0,
                         key=key, format_func=picker_labels.__getitem__, disabled=len(names) == 1,
                         help="各架構的 18 組參考參數依固定 10 題關鍵字查詢的單用詞向量平均 nDCG@10 標註最高與最低。")


@st.cache_data(show_spinner=False)
def search_results(query, signature):
    engine = resources(signature)[0]
    return {method: engine.rank(query, method) for method in METHODS}


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def report_resources(signature):
    return load_analysis()


def report_downloads(key, compact=False):
    signature = analysis_signature()
    if not signature:
        return
    _, path = report_resources(signature)
    st.download_button("下載完整整合報告", (path / "HW4_REPORT.md").read_bytes(),
                       "HW4_REPORT.md", "text/markdown", key=f"{key}_report")
    if not compact:
        st.download_button("下載報告與數據證據包", (path / "HW4_REPORT.zip").read_bytes(),
                           "HW4_REPORT.zip", "application/zip", key=f"{key}_bundle")


@st.cache_resource(show_spinner=False, max_entries=1)
def training_pair_inputs(signature):
    training = [doc for doc in resources(signature)[2] if doc["split"] == "train"]
    return training, [Counter(terms(document_text(doc))) for doc in training]


def find_training_sentence(word, term, training, counts, news_ids=None, max_distance=None):
    for doc, count in zip(training, counts):
        if news_ids is not None and doc["id"] not in news_ids:
            continue
        if word not in count or (term is not None and term not in count):
            continue
        for block in document_blocks(doc):
            for sentence in split_sentences(block):
                tokens = terms(sentence.text)
                if word not in tokens or (term is not None and term not in tokens):
                    continue
                example = {"id": doc["id"], "title": doc["title"], "text": sentence.text}
                if term is None:
                    return example
                anchor_positions = [i for i, token in enumerate(tokens) if token == word]
                neighbor_positions = [i for i, token in enumerate(tokens) if token == term]
                distance = min(abs(a - b) for a in anchor_positions for b in neighbor_positions)
                if max_distance is None or distance <= max_distance:
                    return {**example, "distance": distance}
    return None


@st.cache_data(show_spinner=False, max_entries=512)
def training_example_sentence(word, term, signature, news_ids=(), max_distance=None):
    training, counts = training_pair_inputs(signature)
    if news_ids:
        example = find_training_sentence(word, term, training, counts, news_ids, max_distance)
        if example:
            return example
    return find_training_sentence(word, term, training, counts, max_distance=max_distance)


@st.cache_resource(show_spinner=False, max_entries=1)
def training_sentence_tokens(signature):
    manifest = resources(signature)[1]
    path = RUN / "training_sentences.jsonl"
    if sha256(path) != manifest["artifacts_sha256"][path.name]:
        raise ValueError("訓練句子與模型所記錄的版本不符。")
    sentences = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sentences.append((row["id"], row["tokens"]))
    return sentences


@st.cache_data(show_spinner=False, max_entries=256)
def nearby_word_counts(word, signature):
    """Count training documents that place each candidate near this word."""
    by_document = {}
    for doc_id, tokens in training_sentence_tokens(signature):
        if word not in tokens:
            continue
        nearby = by_document.setdefault(doc_id, set())
        for index, token in enumerate(tokens):
            if token == word:
                start = max(0, index - CONTEXT_RADIUS)
                nearby.update(tokens[start:index + CONTEXT_RADIUS + 1])
    return Counter(token for nearby in by_document.values() for token in nearby if token != word)


def notable_context_words(counts, space, excluded, minimum=1):
    candidates = [word for word, count in counts.items()
                  if count >= minimum and word not in excluded and any(char.isalpha() for char in word)]
    return sorted(candidates, key=lambda word: (-counts[word] * space.idf[space.positions[word]], word))[:2]


def shared_context_words(left, right, space, excluded):
    return notable_context_words(left & right, space, excluded, minimum=2)


@st.cache_data(show_spinner=False, max_entries=512)
def corpus_neighbor_insight(word, term, signature, note=None):
    training, counts = training_pair_inputs(signature)
    evidence = pair_evidence(word, term, training, counts)
    if note:
        doc, count = next(((doc, count) for doc, count in zip(training, counts) if doc["id"] == note["news"]),
                          (None, {}))
        if word not in count or term not in count:
            raise ValueError(f"補充解讀缺少對應的訓練新聞原文。{word}／{term}")
        witness = find_training_sentence(word, term, (doc,), (count,))
        evidence["witnesses"] = [witness] if witness else excerpt(doc, (word, term))
        return {**note, "source": "補充逐詞解讀", "evidence": evidence}

    witness = find_training_sentence(word, term, training, counts) if evidence["cooccurrence_documents"] else None
    if witness:
        evidence["witnesses"] = [witness]
    left, right = nearby_word_counts(word, signature), nearby_word_counts(term, signature)
    space = resources(signature)[0].space
    excluded = {word, term}
    shared = shared_context_words(left, right, space, excluded)
    if shared:
        relation = f"共同用詞 {shared[0]}"
        uses = "、".join(f"{token}（{left[token]}／{right[token]} 篇）" for token in shared)
        reading = f"兩詞周圍都可見 {uses}。篇數依 {word}、{term} 的順序列出。"
        questions = ["context"]
    else:
        relation = "各自的周邊用詞"
        anchor_context = "、".join(notable_context_words(left, space, excluded)) or "這段範圍的詞很少"
        neighbor_context = "、".join(notable_context_words(right, space, excluded)) or "這段範圍的詞很少"
        reading = f"可對照 {word} 周圍的用詞（{anchor_context}），以及 {term} 周圍的用詞（{neighbor_context}）。"
        questions = ["unexpected"]
    return {"relation": relation, "reading": reading, "source": CORPUS_SOURCE,
            "questions": questions, "evidence": evidence}


def neighbor_insights(word, entry, points_by_model, signature):
    context_terms = CONTEXT_NEIGHBORS.get(word, ())
    annotations = entry["annotations"] if entry else {}
    evidence = {row["neighbor"]: row for row in entry["evidence"]} if entry else {}
    shown = dict.fromkeys(point["word"] for points in points_by_model.values() for point in points)
    supplemental = {}
    if word in ANALYSIS_WORDS:
        notes = read_json(ROOT / "resources" / "neighbor_context_notes.json")
        if notes["corpus_sha256"] != resources(signature)[3]["documents_sha256"]:
            raise ValueError("補充逐詞解讀的 Reuters 語料版本不符。")
        supplemental = notes["words"].get(word, {})
    insights = {}
    for term in shown:
        note = supplemental.get(term)
        if term not in annotations or (note and "meaning" in note):
            insights[term] = corpus_neighbor_insight(word, term, signature, note)
            continue
        annotation = annotations[term]
        unexpected = annotation.get("unexpected", False)
        flags = {"reasonable": not unexpected, "context": term in context_terms, "unexpected": unexpected}
        insights[term] = {**annotation, "source": "逐詞解讀", "questions": [key for key in WORD_QUESTIONS if flags[key]],
                          "evidence": evidence[term]}
    return insights


def pair_training_sentence(word, term, evidence, signature):
    witnesses = evidence["witnesses"]
    sentence = next((row for row in witnesses if "distance" in row), None)
    if sentence is None and evidence["cooccurrence_documents"]:
        news_ids = tuple(dict.fromkeys(row["id"] for row in witnesses))
        sentence = training_example_sentence(word, term, signature, news_ids)
    return sentence


def show_training_sentence(example, highlights):
    st.caption(f"原文例句 · 訓練新聞 #{example['id']}")
    original = example["text"]
    parts = []
    start = 0
    for match in TOKEN_RE.finditer(original):
        if normalize(match.group()) in highlights:
            parts.append(escape(original[start:match.start()]))
            parts.append(f"<mark>{escape(match.group())}</mark>")
            start = match.end()
    parts.append(escape(original[start:]))
    st.html(f'<p style="white-space: pre-wrap; overflow-wrap: anywhere">{"".join(parts)}</p>')


def show_pair_examples(word, term, evidence, signature):
    with st.expander(f"查看 {term} 的原文例句"):
        sentence = pair_training_sentence(word, term, evidence, signature)
        if sentence:
            show_training_sentence(sentence, (word, term))
            return

        news_ids = tuple(dict.fromkeys(row["id"] for row in evidence["witnesses"]))
        for col, focus in zip(st.columns(2, gap="medium"), (word, term)):
            with col:
                example = training_example_sentence(focus, None, signature, news_ids)
                if example:
                    st.markdown(f"**`{focus}` 的用法**")
                    show_training_sentence(example, (focus,))


def show_shared_contexts(word, term, insight, signature):
    counts = {focus: nearby_word_counts(focus, signature) for focus in (word, term)}
    space = resources(signature)[0].space
    shared = shared_context_words(counts[word], counts[term], space, {word, term})
    if not shared:
        return False
    with st.expander(f"查看 {term} 的共同用詞例句（{'、'.join(shared)}）"):
        for token in shared:
            st.markdown(f"**共同用詞 `{token}`**")
            for col, focus in zip(st.columns(2, gap="medium"), (word, term)):
                with col:
                    st.markdown(f"**在 `{focus}` 旁邊** · {counts[focus][token]} 篇訓練新聞")
                    example = training_example_sentence(focus, token, signature, max_distance=CONTEXT_RADIUS)
                    if example:
                        show_training_sentence(example, (focus, token))
                    else:
                        st.caption("這一欄暫無可核對的原文例句。")
            reading = insight.get("context_readings", {}).get(token)
            if reading:
                st.caption(reading)
    return True


def show_related_neighbor(word, term, insight, signature, context=False):
    evidence = insight["evidence"]
    st.markdown(f"#### `{term}`")
    reading = f"{insight['relation']}。{insight['reading']}"
    if context and insight["source"] == CORPUS_SOURCE:
        reading = "兩個詞的周圍反覆出現相同的用詞，這些原句可用來對照它們的用法。"
    st.markdown(f"**和 `{word}` 的關係**　{reading}")
    st.caption(f"`{term}` 見於 {evidence['neighbor_df']} 篇訓練新聞，"
               f"與 `{word}` 同篇共現 {evidence['cooccurrence_documents']} 篇。")
    if context:
        if show_shared_contexts(word, term, insight, signature):
            return
        st.caption("達到兩邊各至少兩篇門檻的共同用詞為 0 個。展開例句可對照新聞情境。")
    show_pair_examples(word, term, evidence, signature)


def show_unexpected_neighbor(word, term, insight, windows, signature):
    evidence = insight["evidence"]
    witnesses = evidence["witnesses"]
    cooccurrence = evidence["cooccurrence_documents"]
    frequency = evidence["neighbor_df"]
    sentence = pair_training_sentence(word, term, evidence, signature)

    reasons = []
    if "meaning" in insight:
        meaning, surprise = insight["meaning"], insight["surprise"]
        reasons.append(insight["reading"])
    elif insight["source"] != CORPUS_SOURCE:
        meaning = f"{insight['relation']}。{insight['reading']}"
        surprise = f"以 `{word}` 找常見的近義詞或搭配時，這種特定的新聞用法排進了前五名。"
    else:
        meaning = ("原文中的數字，片段保留了它原來的寫法。" if term.replace(".", "", 1).isdecimal()
                   else "可展開原文片段，看看這個詞在新聞裡的用法。")
        surprise = ("這個詞排進了前五名。查看兩詞左右各兩個詞時，共同用詞的證據較少，"
                    "這個高排名值得回到原文檢查。")

    if sentence:
        distance = sentence["distance"]
        matching = "、".join(f"{model} 的窗口 {window}" for model, window in windows.items() if distance <= window)
        clue = f"兩詞在原文例句同句出現，前處理後相距 {distance} 個詞位。"
        if matching:
            clue += f"這個距離在 {matching} 可涵蓋的範圍內，訓練時有機會把兩詞一起納入上下文。"
        else:
            clue += "這個距離超出目前所選模型的窗口，例句可用來了解它們的新聞情境。"
        reasons.append(clue)
    elif cooccurrence:
        reasons.append("原文片段顯示兩詞出現在同一篇新聞，可能因共同的報導主題而靠近。")
    else:
        reasons.append("兩詞的原文片段來自不同新聞。各自的句型或報導情境可用來找相似用法。")
    if 0 < frequency <= 2:
        reasons.append(f"`{term}` 只見於 {frequency} 篇訓練新聞，少數報導的用法可能對詞向量有較大影響。")

    st.markdown(f"#### `{term}`")
    st.markdown(f"**在原文裡是什麼**　{meaning}\n\n"
                f"**為什麼排進前五名會讓人意外**　{surprise}\n\n"
                f"**可能靠近 `{word}` 的原因**　{''.join(reasons)}")
    st.caption(f"`{term}` 見於 {frequency} 篇訓練新聞，與 `{word}` 同篇共現 {cooccurrence} 篇。")
    label = "原文例句" if sentence else "原文片段"
    examples = [sentence] if sentence else witnesses
    if examples:
        with st.expander(f"查看 {term} 的{label}"):
            for witness in examples:
                st.markdown(f"**{label} · 訓練新聞 #{witness['id']}**")
                st.text(witness["text"])


def word_questions(word, points_by_model, insights, configs, signature):
    st.subheader(f"{word} 的近鄰詞解讀")
    st.caption("同一詞對可對應多題。周邊用詞統計比較同一句內、目標詞左右各兩個詞的用法。")
    for number, (key, question) in enumerate(WORD_QUESTIONS.items(), 1):
        with st.container(border=True):
            st.markdown(f"**{number}. {question}**")
            if key == "context":
                st.caption("在前處理後的同一句中，從兩詞左右各兩個詞的範圍找共同用詞。"
                           "共同用詞須在兩邊各見於至少兩篇訓練新聞，每欄各列一個原句。")
            terms_with_evidence = {}
            for col, (model, label) in zip(st.columns(2), WORD_MODELS.items()):
                matches = [point for point in points_by_model[model]
                           if key in insights[point["word"]]["questions"]]
                ranks = "、".join(f"`{point['word']}` 第 {point['rank']} 名" for point in matches)
                col.markdown(f"**{label}**  \n{ranks or '無此類例子'}")
                terms_with_evidence.update(dict.fromkeys(point["word"] for point in matches))
            for term in terms_with_evidence:
                if key == "unexpected":
                    windows = {WORD_MODELS[model]: configs[model]["window"] for model, points in points_by_model.items()
                               if any(point["word"] == term for point in points)}
                    show_unexpected_neighbor(word, term, insights[term], windows, signature)
                else:
                    show_related_neighbor(word, term, insights[term], signature, context=key == "context")


def word_interpretation(word, points_by_model, insights, widget_suffix, evidence_terms):
    st.subheader(f"{word} 的逐詞解讀")
    options = list(insights)
    default = next((term for term in evidence_terms if term in insights), options[0])
    term = st.selectbox("選擇近鄰核對 Reuters 原文", options, index=options.index(default),
                        format_func=lambda term: f"{term} · {insights[term]['relation']}",
                        key=f"neighbor_detail_{widget_suffix}")
    annotation = insights[term]
    row = annotation["evidence"]
    st.markdown(f"**{word} ↔ {term} · {annotation['relation']}**")
    ranks = []
    for model, label in WORD_MODELS.items():
        match = next((point for point in points_by_model[model] if point["word"] == term), None)
        if match:
            ranks.append(f"{label} 第 {match['rank']} 名 · cosine {match['cosine']:.4f}")
    st.caption(" ｜ ".join(ranks))
    st.caption(f"判讀依據 · {annotation['source']}")
    st.write(annotation["reading"])
    st.caption(f"訓練 CF {row['neighbor_cf']} 次 · DF {row['neighbor_df']} 篇 · 與 {word} 共現 {row['cooccurrence_documents']} 篇。"
               "每篇同時含有兩個詞的訓練新聞計為 1 篇。")
    with st.expander("原文證據 · Reuters 訓練新聞"):
        if row["cooccurrence_documents"] == 0:
            st.caption("兩詞沒有同篇紀錄。以下原文分別顯示各自的用法。")
        for witness in row["witnesses"]:
            st.markdown(f'**訓練新聞 {witness["id"]} · {witness["title"]}**')
            st.text(witness["text"])
        st.caption("原文與統計提供語境線索。共現篇數是文件統計，Word2Vec 的窗口共現需要另算。")
    report_downloads("words", compact=True)


def document_case_panel(engine, selected_qid=None):
    signature = analysis_signature()
    if not signature:
        return
    payload, _ = report_resources(signature)
    by_qid = {case["query"]["qid"]: case for case in payload["cases"]}
    st.divider()
    st.subheader("文件排名案例解讀")
    # Follow a directly selected case query once; the reader may then explore other cases.
    if st.session_state.get("case_query_seed") != selected_qid:
        if selected_qid in by_qid:
            st.session_state["case_qid"] = selected_qid
        st.session_state["case_query_seed"] = selected_qid
    qid = st.selectbox("選擇案例", list(by_qid), key="case_qid",
                       format_func=lambda q: f'{q} · {by_qid[q]["title"]}')
    case = by_qid[qid]
    st.markdown(f'**{case["kind"]}｜{case["query"]["text"]}**')
    st.caption(f'目標 topic：{case["query"]["topic"]}。以下名次來自三方法的完整排名。')
    frame = pd.DataFrame([{"方法": METHODS[row["method"]], "P@10": row["p10"], "RR": row["mrr"],
                           "nDCG@10": row["ndcg10"]} for row in case["metrics"]])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.markdown("**已觀察到的事實**")
    for text in case["observations"]:
        st.write("• " + text)
    st.markdown("**可能原因**")
    st.write(case["interpretation"])
    st.markdown("**可推論範圍與標籤限制**")
    st.write(case["limitation"])
    with st.expander("查看新聞名次、分數與原文", expanded=False):
        rows = [{"新聞 ID": row["id"], "標題": row["title"], "topics": "、".join(row["topics"]),
                 "目標 topic 命中": row["topic_relevant"],
                 "共同查詢詞": "、".join(row["shared_query_terms"]),
                 **{METHODS[method] + " 名次": row["ranks"][method] for method in METHODS},
                 **{METHODS[method] + " cosine": row["scores"][method] for method in METHODS}}
                for row in case["documents"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        keys = [row["id"] for row in case["documents"]]
        selected = st.selectbox("閱讀案例新聞", keys, key="case_document",
                                format_func=lambda key: f'{key} · {engine.by_id[key]["title"][:100]}')
        if not engine.by_id[selected]["body"]:
            st.caption("此筆原始資料僅有標題，沒有正文；分析未補造內容。")
        read_news(engine.by_id[selected])


def heading(kicker, title, description):
    st.caption(kicker)
    st.title(title)
    st.write(description)


def read_news(doc):
    st.subheader(doc["title"] or "（無標題）")
    labels = "、".join(doc["topics"]) or "無主題標籤"
    st.caption(f'新聞 {doc["id"]}　·　{doc["date"]}　·　topics：{labels}')
    content = doc["body"] or doc["title"] or "此筆新聞沒有可用文字。"
    st.text_area("新聞全文", content, height=310, disabled=True, key=f'fulltext_{doc["id"]}')


def overview(engine, manifest, documents, data_manifest):
    heading("01 / CORPUS", "Reuters 語料概況", "以固定的新聞集合，觀察詞彙表示如何影響搜尋結果。")
    for col, label, value in zip(st.columns(4), ["訓練新聞", "測試新聞", "訓練詞彙", "Word2Vec 模型"],
                                 [9603, 3299, manifest["train_vocabulary"], 2]):
        col.metric(label, f"{value:,}")
    st.caption("模型、詞表與 IDF 僅使用訓練集；搜尋與評估涵蓋完整測試集。另 8,676 篇 unused 新聞未參與訓練或排名。")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("五個分析主題")
        counts = [{"主題": f"{topic} · {label}", "切分": split, "新聞數": sum(topic in d["topics"] for d in documents if d["split"] == key)}
                  for topic, label in TOPICS.items() for key, split in (("train", "訓練集"), ("test", "測試集"))]
        chart = alt.Chart(pd.DataFrame(counts)).mark_bar(cornerRadiusTopRight=3).encode(
            y=alt.Y("主題:N", sort=list(dict.fromkeys(r["主題"] for r in counts)), title=None),
            x=alt.X("新聞數:Q", title="新聞篇數"), yOffset="切分:N",
            color=alt.Color("切分:N", scale=alt.Scale(range=COLORS[:2]), legend=alt.Legend(orient="bottom")),
            tooltip=["主題", "切分", "新聞數"]).properties(height=300)
        st.altair_chart(chart, width="stretch")
        st.caption("一篇新聞可以有多個 topics，因此各主題篇數會重疊。其他主題與空 topics 的測試新聞也保留為搜尋候選。")
    with right:
        st.subheader("三種文件表示")
        st.markdown("**TF-IDF**　以詞項重要性建立稀疏向量，依共同詞彙匹配。")
        st.markdown("**TF-IDF 加權 CBOW**　用上下文預測中心詞，將訓練完成的詞向量加權成新聞向量。")
        st.markdown("**TF-IDF 加權 Skip-Gram**　用中心詞預測上下文，以相同方式表示新聞與查詢。")
        st.info("三種方法使用相同查詢、前處理及測試新聞，全部以原維度 cosine 排名。")
    with st.expander("計算方式與訓練設定"):
        st.latex(r"\mathrm{IDF}(t)=\log_{10}\frac{N_{\mathrm{train}}}{\mathrm{DF}_{\mathrm{train}}(t)},\qquad w(t,d)=\mathrm{TF}(t,d)\,\mathrm{IDF}(t)")
        st.latex(r"\mathbf{v}(d)=\frac{\sum_{t\in d\cap V_{\mathrm{w2v}}}w(t,d)\,\mathbf{e}(t)}{\sum_{t\in d\cap V_{\mathrm{w2v}}}w(t,d)}")
        st.write("查詢使用相同公式並保留重複詞 TF。NFC／小寫化、固定停用詞、不 stemming；新聞使用標題＋正文，topics 不進入向量。OOV 忽略，零向量不補造排名。")
        st.write("兩模型固定 100 維、window=5、min_count=2、30 epochs、seed=42、workers=1、negative=5。CBOW 與 Skip-Gram 各自從頭訓練，沒有使用預訓練權重。")
        st.write("訓練流程示意：在 crude oil prices 這個示意序列中，CBOW 可用 crude／prices 預測中心詞 oil；Skip-Gram 可由 oil 預測 crude／prices。正式實驗使用訓練新聞、window=5，這個示例不會加入語料。")
    st.divider()
    tab1, tab2 = st.tabs(["瀏覽測試新聞", "TF／DF／IDF 詞項統計"])
    with tab1:
        topic = st.selectbox("瀏覽主題", ["全部", *TOPICS], key="browse_topic")
        choices = [d for d in engine.documents if topic == "全部" or topic in d["topics"]]
        if choices:
            selected = st.selectbox("選擇新聞", [d["id"] for d in choices],
                                    format_func=lambda key: f'{key} · {engine.by_id[key]["title"][:110]}', key="browse_document")
            st.caption(f"符合瀏覽條件：{len(choices):,} 篇。此篩選只影響閱讀，不改變搜尋與評估候選集合。")
            read_news(engine.by_id[selected])
    with tab2:
        text = st.text_input("搜尋詞項", placeholder="例如 oil、profit", key="term_filter").strip().casefold()
        space = engine.space
        indices = [i for i, word in enumerate(space.vocabulary) if text in word]
        indices.sort(key=lambda i: (-int(space.cf[i]), space.vocabulary[i]))
        frame = pd.DataFrame([{"詞項": space.vocabulary[i], "TF（訓練集總次數）": int(space.cf[i]),
                               "DF（訓練新聞篇數）": int(space.df[i]), "IDF": float(space.idf[i])} for i in indices[:100]])
        st.caption(f"符合 {len(indices):,} 詞，顯示總 TF 最高的前 100 詞。每篇新聞計分仍使用該篇的 TF。")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("下載完整詞項統計", (RUN / "term_statistics.csv").read_bytes(), "term_statistics.csv", "text/csv")
    with st.expander("資料來源與版本"):
        st.markdown(f"[Reuters-21578 原始資料說明]({README_URL})")
        st.caption(f'語料 SHA256：{data_manifest["documents_sha256"]}')
        st.download_button("下載資料清單", json.dumps(data_manifest, ensure_ascii=False, indent=2), "corpus_manifest.json")


@st.cache_data(show_spinner=False)
def training_context(word, signature):
    documents = resources(signature)[2]
    contexts = []
    for doc in documents:
        if doc["split"] == "train" and word in terms(document_text(doc)):
            content = document_text(doc)
            position = normalize(content).find(word)
            contexts.append({"id": doc["id"], "title": doc["title"],
                             "excerpt": content[max(0, position - 100):position + 240].replace("\n", " ")})
            if len(contexts) == 2:
                break
    return contexts


def neighbor_context(term, insights):
    insight = insights.get(term)
    if insight is None:
        return {"relation": "中心詞", "cooccurrence": None, "reading": None, "source": None}
    return {
        "relation": insight["relation"],
        "cooccurrence": insight["evidence"]["cooccurrence_documents"],
        "reading": insight["reading"],
        "source": insight["source"],
    }


def insight_tooltips():
    return [alt.Tooltip("relation:N", title="語境關係"), alt.Tooltip("source:N", title="判讀依據"),
            alt.Tooltip("cooccurrence:Q", title="同篇共現（篇）"), alt.Tooltip("reading:N", title="新聞用法")]


def neighbor_similarity_chart(word, points, color, insights):
    frame = pd.DataFrame([
        {**point, "label": f"{point['rank']} · {point['word']}", **neighbor_context(point["word"], insights)}
        for point in points
    ])
    base = alt.Chart(frame).encode(
        x=alt.X("cosine:Q", title=f"與 {word} 的原始 cosine",
                scale=alt.Scale(domain=[-1, 1], nice=False, padding=44),
                axis=alt.Axis(values=[-1, -.5, 0, .5, 1], format=".1f", domain=False,
                              ticks=False, labelPadding=8, titlePadding=12, gridColor="#DCE5E1")),
        y=alt.Y("label:N", sort=frame["label"].tolist(), title=None,
                axis=alt.Axis(domain=False, ticks=False, labelOverlap=False, labelPadding=10,
                              labelFontSize=12, labelLimit=140, minExtent=100)),
    )
    bars = base.mark_bar(color=color, size=15, cornerRadiusEnd=4).encode(
        tooltip=[alt.Tooltip("rank:Q", title="名次"), alt.Tooltip("word:N", title="鄰近詞"),
                 alt.Tooltip("cosine:Q", title="原始 cosine", format=".6f"),
                 *insight_tooltips()],
    )
    labels = base.mark_text(
        align=alt.ExprRef(expr="datum.cosine < 0 ? 'right' : 'left'"),
        dx=alt.ExprRef(expr="datum.cosine < 0 ? -6 : 6"),
        baseline="middle", fontSize=12, color="#173042",
    ).encode(text=alt.Text("cosine:Q", format=".4f"))
    return (bars + labels).properties(
        height=alt.Step(36), autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
    ).configure_view(strokeOpacity=0)


cached_neighbor_projection = st.cache_data(show_spinner=False, max_entries=128)(project_neighbor_space)


@st.cache_data(show_spinner="正在準備全部詞的投影…", max_entries=12)
def project_vocabulary(model_signature, architecture, method, _vectors):
    return get_vocabulary_projection(model_signature, architecture, method, _vectors)


def projection_labels(frame, extent, side):
    """Place text on the fixed square view without moving projected points."""
    scale = side / (2 * extent)
    marked = [(row.label, (row.x + extent) * scale, (row.y + extent) * scale)
              for row in frame.sort_values("rank", kind="stable").itertuples()]
    occupied = [(x - 9, y - 9, x + 9, y + 9) for _, x, y in marked]
    labels = []
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0))
    for term, x, y in marked:
        width, height = min(130, 8 * len(term) + 6), 17
        chosen = None
        for gap in (8, 14, 22, 33, 49, 68, 92):
            for dx, dy in directions:
                left = np.clip(x + dx * gap - width * (1 - dx) / 2, 2, side - width - 2)
                bottom = np.clip(y + dy * gap - height * (1 - dy) / 2, 2, side - height - 2)
                right, top = left + width, bottom + height
                if not any(left < r and right > l and bottom < t and top > b for l, b, r, t in occupied):
                    chosen = (left, bottom, right, top)
                    break
            if chosen is not None:
                break
        if chosen is not None:
            left, bottom, right, top = chosen
            occupied.append(chosen)
            labels.append({"label": term, "label_x": (left + 2) / scale - extent,
                           "label_y": (bottom + height / 2) / scale - extent,
                           "x": x / scale - extent, "y": y / scale - extent,
                           "end_x": np.clip(x, left, right) / scale - extent,
                           "end_y": np.clip(y, bottom, top) / scale - extent})
    return pd.DataFrame(labels, columns=["label", "label_x", "label_y", "x", "y", "end_x", "end_y"])


def tsne_topic_regions(words, xy, extent):
    """Choose up to three separated regions supported by nearby topic examples."""
    coordinates = dict(zip(words, xy, strict=True))
    candidates = []
    for topic, examples in TSNE_TOPIC_WORDS.items():
        terms = [term for term in examples if term in coordinates]
        if len(terms) < 3:
            continue
        locations = np.array([coordinates[term] for term in terms])
        distances = np.linalg.norm(locations[:, None] - locations[None, :], axis=2)
        nearby = distances <= .13 * extent
        seed = max(range(len(terms)), key=lambda i: (nearby[i].sum(), -distances[i, nearby[i]].sum()))
        members = nearby[seed]
        if members.sum() < 3:
            continue
        center = locations[members].mean(axis=0)
        radius = float(np.linalg.norm(locations[members] - center, axis=1).max()) + .025 * extent
        radius = max(radius, .06 * extent)
        if np.abs(center).max() + radius <= extent:
            candidates.append({"topic": topic, "center": center, "radius": radius,
                               "terms": tuple(term for term, keep in zip(terms, members) if keep)})

    selected = []
    for region in sorted(candidates, key=lambda item: (-len(item["terms"]), item["radius"])):
        if any(np.linalg.norm(region["center"] - other["center"]) <= region["radius"] + other["radius"] + .07 * extent
               for other in selected):
            continue
        selected.append(region)
        if len(selected) == 3:
            break
    return selected


def tsne_topic_layers(regions, frame, extent, x, y, tooltips):
    color = "#744493"
    angles = np.linspace(0, 2 * np.pi, 49)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    rings, titles, topics_by_term = [], [], {}
    for region in regions:
        topic, center, radius = region["topic"], region["center"], region["radius"]
        for step, (cx, cy) in enumerate(center + radius * directions):
            rings.append({"topic": topic, "step": step, "x": cx, "y": cy})
        titles.append({"topic": topic, "x": np.clip(center[0], -.74 * extent, .74 * extent),
                       "y": min(center[1] + radius, .89 * extent)})
        topics_by_term.update(dict.fromkeys(region["terms"], topic))

    outline = alt.Chart(pd.DataFrame(rings)).mark_line(
        color=color, strokeWidth=1.6, strokeDash=[5, 3], opacity=.9, clip=True, aria=False,
    ).encode(x=x, y=y, detail="topic:N", order="step:Q")
    examples = frame.loc[frame["word"].isin(topics_by_term) & (frame["rank"] > 5)].copy()
    examples["topic"] = examples["word"].map(topics_by_term)
    example_points = alt.Chart(examples).mark_point(filled=True, color=color, size=38, opacity=.95, clip=True).encode(
        x=x, y=y, tooltip=[*tooltips, alt.Tooltip("topic:N", title="代表詞群")],
    )
    labels = alt.Chart(pd.DataFrame(titles)).encode(x=x, y=y, text="topic:N")
    label_style = dict(fontSize=12, fontWeight="bold", baseline="bottom", dy=-4, clip=True)
    halo = labels.mark_text(color="#F7F7F3", stroke="#F7F7F3", strokeWidth=3, aria=False, **label_style)
    return outline, example_points, halo, labels.mark_text(color=color, **label_style)


def word_space_chart(words, xy, cosine, extent, color, method, insights, *, pan_zoom=False, topic_regions=()):
    side = 304
    dense = pan_zoom and len(words) > 6
    if dense:
        ranks = np.arange(len(words), dtype=np.int32)
        frame = pd.DataFrame({"word": words, "x": xy[:, 0], "y": xy[:, 1], "cosine": cosine,
                              "rank": ranks})
        labeled = frame.iloc[:6].assign(
            kind=["中心詞", *["近鄰"] * 5],
            label=[words[0], *(f"{rank} · {term}" for rank, term in enumerate(words[1:6], 1))],
        )
    else:
        frame = pd.DataFrame([
            {"word": term, "x": xy[i, 0], "y": xy[i, 1], "cosine": cosine[i],
             "rank": i, "label": f"{i} · {term}" if i else term, "kind": "近鄰" if i else "中心詞",
             **(neighbor_context(term, insights) if i <= 5 else {})}
            for i, term in enumerate(words)
        ])
        labeled = frame.iloc[:6]
    pca = method == "PCA"
    x_title, y_title = ("X（PC1）", "Y（PC2）") if pca else ("X（t-SNE 1）", "Y（t-SNE 2）")
    scale = alt.Scale(domain=[-extent, extent], nice=False, zero=False)
    axis_style = dict(tickCount=5, domain=True, labels=True, ticks=True, grid=True,
                      gridColor="#DCE5E1", format=",.2~f", labelPadding=4, titlePadding=9)
    x_axis = alt.Axis(title=x_title, **axis_style)
    y_axis = alt.Axis(title=y_title, **axis_style)
    x = alt.X("x:Q", scale=scale, axis=x_axis)
    y = alt.Y("y:Q", scale=scale, axis=y_axis)
    base = alt.Chart(labeled if dense else frame).encode(x=x, y=y)
    tooltips = [alt.Tooltip("label:N", title="詞與名次"), alt.Tooltip("kind:N", title="圖例"),
                alt.Tooltip("cosine:Q", title="與中心詞的原始 cosine", format=".6f")]
    if pca:
        tooltips.extend([alt.Tooltip("x:Q", title="PC1", format=".4f"), alt.Tooltip("y:Q", title="PC2", format=".4f")])
    if len(words) <= 6:
        tooltips.extend(insight_tooltips())
    symbols = base.mark_point(filled=True, color=color, clip=pan_zoom).encode(
        shape=alt.Shape("kind:N", scale=alt.Scale(domain=["中心詞", "近鄰"],
                                                  range=["square", "circle"]), legend=None),
        size=alt.condition(alt.datum.rank <= 5, alt.value(100), alt.value(8)),
        opacity=alt.condition(alt.datum.rank <= 5, alt.value(1), alt.value(.22)),
        order=alt.Order("rank:Q", sort="descending"),
        tooltip=tooltips,
    )
    if dense:
        tooltips = [alt.Tooltip("word:N", title="詞"), alt.Tooltip("rank:Q", title="名次"), *tooltips[2:]]
        # A constant-style layer avoids computing a description and highlight style for every point.
        cloud = alt.Chart(frame).transform_filter(alt.datum.rank > 5).mark_point(
            filled=True, shape="circle", color=color, size=8, opacity=.22, clip=True, aria=False,
        ).encode(x=x, y=y, tooltip=tooltips).interactive(name="projection_view")
        points = [cloud, symbols]
    else:
        points = [symbols.interactive(name="projection_view") if pan_zoom else symbols]
    labels = alt.Chart(projection_labels(labeled, extent, side))
    leaders = labels.mark_rule(color="#839194", opacity=.6, clip=pan_zoom).encode(
        x=x, y=y, x2="end_x:Q", y2="end_y:Q")
    text = labels.mark_text(align="left", fontSize=12, limit=123, clip=pan_zoom).encode(
        x=alt.X("label_x:Q", scale=scale, axis=x_axis), y=alt.Y("label_y:Q", scale=scale, axis=y_axis), text="label:N")
    layers = [leaders, *points, text]
    if topic_regions:
        outline, example_points, halo, group_labels = tsne_topic_layers(topic_regions, frame, extent, x, y, tooltips)
        layers = [leaders, *points, outline, example_points, text, halo, group_labels]
    chart = alt.layer(*layers).properties(
        width=side, height=side, autosize=alt.AutoSizeParams(type="pad", contains="content"),
    ).configure_view(strokeOpacity=0)
    if pan_zoom:
        chart = chart.properties(usermeta={"embedOptions": {"renderer": "canvas"}})
    return chart


def word_projection_panel(model, word, points, method, color, insights, widget_suffix, *, full_projection=None):
    vocabulary, vectors = model
    words = [word, *(point["word"] for point in points)]
    positions = {term: index for index, term in enumerate(vocabulary)}
    indices = [positions[term] for term in words]
    projection = full_projection or cached_neighbor_projection(vectors[indices], method)
    xy = projection["xy"]
    if full_projection is not None and xy is not None:
        xy = xy[indices]
    pca = method == "PCA"
    if xy is None:
        st.info("t-SNE 需要至少 3 個不同的正規化向量。這組詞可用 PCA 或 Top-5 相似度查看。")
    else:
        extent = max(1., float(np.abs(xy).max()) * 1.12)
        if pca:
            pc1, pc2 = projection["variance_ratio"]
            var1, var2 = projection["variance"]
            st.caption(f"{len(words):,} 個詞 · X（PC1）**{pc1:.1%}** · Y（PC2）**{pc2:.1%}** · 合計 {pc1 + pc2:.1%}",
                       help=f"正規化向量投影後的樣本變異量：X（PC1）{var1:.4f}，Y（PC2）{var2:.4f}。")
        else:
            st.caption(f"{len(words):,} 個詞 · perplexity {projection['perplexity']:.2f} · seed 42")
        cosine = [1., *(point["cosine"] for point in points)]
        topic_regions = tsne_topic_regions(words, xy, extent) if full_projection is not None and not pca else ()
        chart = word_space_chart(words, xy, cosine, extent, color, method, insights,
                                 pan_zoom=full_projection is not None, topic_regions=topic_regions)
        st.altair_chart(chart, width="content")
        if topic_regions:
            with st.expander("查看圈內代表詞"):
                for region in topic_regions:
                    examples = "、".join(f"`{term}`" for term in region["terms"])
                    st.markdown(f"**{region['topic']}**　{examples}")
    projection_pair_panel(words, positions, vectors, xy, pca, widget_suffix, lazy=full_projection is not None)


@st.fragment
def projection_pair_panel(words, positions, vectors, xy, pca, widget_suffix, *, lazy):
    first_key = f"pca_first_{widget_suffix}"
    second_key = f"pca_second_{widget_suffix}"
    expander = st.expander("比較這張圖中的兩個詞", key=f"projection_pair_{widget_suffix}" if lazy else None,
                           on_change="rerun" if lazy else "ignore")
    if lazy and not expander.open:
        # Keep an earlier choice when the large dropdowns are hidden.
        for key in (first_key, second_key):
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]
        return
    with expander:
        a, b = st.columns(2)
        if first_key in st.session_state and st.session_state[first_key] not in words:
            st.session_state[first_key] = words[0]
        first = a.selectbox("詞 A", words, key=first_key)
        alternatives = [term for term in words if term != first]
        preferred = {"profit": "loss", "loss": "profit"}.get(words[0])
        default = preferred if preferred in alternatives else alternatives[0]
        if second_key in st.session_state and st.session_state[second_key] not in alternatives:
            st.session_state[second_key] = default
        second = b.selectbox("詞 B", alternatives, index=alternatives.index(default), key=second_key)
        left, right = unit_rows(vectors[[positions[first], positions[second]]].astype(np.float64))
        similarity = float(np.clip(left @ right, -1., 1.))
        full_distance = np.sqrt(max(0., 2 - 2 * similarity))
        distance_line = f"原始 cosine **{similarity:.4f}**　·　{vectors.shape[1]} 維距離 **{full_distance:.4f}**"
        if pca:
            i, j = words.index(first), words.index(second)
            distance_line += f"　·　2 維距離 **{np.linalg.norm(xy[i] - xy[j]):.4f}**"
        st.markdown(distance_line)
        st.caption("完整向量距離由 √(2 − 2 × cosine) 算出。PCA 的 2 維距離量圖上的投影點。")


@st.cache_data(show_spinner=False, max_entries=256)
def neighbor_points(word, method, model_signature, _model):
    ranked = vocabulary_neighbors(word, method, model_signature, _model)
    return [{"rank": rank, **row} for rank, row in enumerate(ranked[:5], 1)]


@st.cache_data(show_spinner=False, max_entries=12)
def vocabulary_neighbors(word, architecture, model_signature, _model):
    return neighbors(*_model, word, k=len(_model[0]) - 1)


def read_weighting_reference():
    try:
        report = read_comparison()
        reference_scores = {(row["run"], row["method"]): row["unweighted_ndcg10"]
                            for row in report["runs"]}
    except (ValueError, OSError, KeyError, TypeError):
        return None, {}
    return report, reference_scores


@st.fragment
def word_map_panel(word, models, model_signatures, points_by_model, insights, configs, selected_runs, tables):
    view = None
    all_words = False
    if any(points_by_model.values()):
        st.subheader(f"{word} 的 Top-5 近鄰")
        st.caption("原始 cosine 決定各模型的近鄰名次。排名表與下方「逐詞解讀」聚焦前 5 名。"
                   "共現篇數計算有提到兩個詞的訓練新聞。")
        view = st.radio("圖表", ("Top-5 相似度", "PCA 空間", "t-SNE 空間"), horizontal=True, key="word_map_view")
        if view == "Top-5 相似度":
            st.caption("各圖只顯示本模型的 Top-5，與各自的排名表對應。"
                       "長條從 0 起算。向右代表正分，向左代表負分。兩張圖使用相同的 cosine 尺度。")
        else:
            scope = st.radio("投影範圍", ("前 5 詞", "全部詞"), horizontal=True, key="word_projection_limit")
            all_words = scope == "全部詞"
            st.caption("方塊是中心詞，前 5 名用較大的圓點標示。將游標停在點上可查看詞、名次和原始 cosine。"
                       "兩張圖各自投影到二維，跨模型比較請讀原始 cosine 與名次。")
            if all_words:
                st.caption("全部詞使用各模型詞彙表中有可用向量的詞。已準備的投影可直接載入，換中心詞也會沿用座標。")
                if view == "t-SNE 空間":
                    st.caption("紫色虛線依同主題代表詞的位置畫圈，每張圖最多三圈。圈內也可能有其他詞，"
                               "可在圖下展開清單核對用來畫圈的代表詞。")
                st.caption("在圖內滾動滑鼠滾輪可縮放，按住左鍵拖曳可平移，雙擊可回到全圖。")
            if view == "PCA 空間":
                with st.expander("PCA 圖怎麼讀？"):
                    st.markdown(
                        "**X（PC1）與 Y（PC2）**　PCA 用詞向量的原有維度組合出兩個新方向。"
                        "PC1 是詞向量分布最散的方向，PC2 是與 PC1 垂直的方向中變異最大的一個。\n\n"
                        "**變異比例**　圖上方的百分比表示各軸保留了這批詞向量的多少變異。"
                        "例如合計 33.4% 表示兩軸保留約三分之一的變異，問號可查看各軸的變異量原值。"
                        "切換「前 5 詞」與「全部詞」時，參與投影的詞不同，方向和比例都可能改變。\n\n"
                        "**圖上的距離**　兩詞在圖上靠近，表示它們沿這兩個方向的位置接近。"
                        "其他維度的差異可能被省略，因此詞群也可能看起來混在一起。"
                        "在圖下展開「比較這張圖中的兩個詞」，可對照 2 維距離與原始 cosine。"
                    )
            else:
                st.markdown(
                    "**為什麼詞群看起來連在一起？**\n\n"
                    "穀物和航運會共用 `cargo`（貨物）這類詞。Word2Vec 依周邊用詞學習，"
                    "所以 `grain`（穀物）和 `vessels`（船隻）都可以和 `cargo` 相近，兩群的近鄰就有了交集。"
                    "t-SNE 儘量保留這些局部關係，所以兩群即使各有一塊密集區，中間仍可能有其他詞的點接起來。"
                )
                with st.expander("看 Reuters 原文與原始 cosine"):
                    st.markdown(
                        "**同一個詞怎麼用在兩個主題**　訓練新聞 #4203 寫到 `cargo of wheat`（裝載的小麥）。"
                        "訓練新聞 #11271 則同時提到靠港船隻 `vessels` 和 `cargo handling equipment`（貨物裝卸設備）。"
                        "預設 CBOW 的原始向量中，`cargo` 與 `grain` 的 cosine 是 **0.8140**，"
                        "與 `vessels` 是 **0.8875**。\n\n"
                        "**全圖怎麼看**　兩個預設模型各有 23,172 個詞，縮到全圖時點也會重疊。"
                        "在「全部詞」可放大紫色圈附近，再用原始 cosine 和近鄰名次核對。\n\n"
                        "**以 `wheat` 為例**　在預設 CBOW（300 維、窗口 15、5 輪）裡，"
                        "原始 cosine 排名第一的是 `corn`（**0.9493**）。在預設 Skip-Gram"
                        "（150 維、窗口 15、15 輪）裡，第一名是 `barley`（**0.7390**）。"
                        "兩者都是預設模型穀物圈的代表詞。\n\n"
                        "[t-SNE 圖的判讀示例](https://distill.pub/2016/misread-tsne/)"
                    )
    for col, (name, label), color in zip(st.columns(2, gap="large"), WORD_MODELS.items(), COLORS):
        with col:
            st.markdown("#### " + label)
            points = points_by_model[name]
            if not points:
                st.info(f"這個詞沒有可用詞向量，可能未出現在訓練集或未達 min_count={configs[name]['min_count']}。")
                continue
            if view == "Top-5 相似度":
                st.altair_chart(neighbor_similarity_chart(word, points, color, insights), width="stretch")
            else:
                method = "PCA" if view == "PCA 空間" else "t-SNE"
                full_projection, plotted_points = None, points
                if all_words:
                    full_projection = project_vocabulary(model_signatures[name], name, method, models[name][1])
                    plotted_points = vocabulary_neighbors(word, name, model_signatures[name], models[name])
                word_projection_panel(models[name], word, plotted_points, method, color, insights,
                                      f"{name}_{word}_{selected_runs[name]}", full_projection=full_projection)
            st.dataframe(tables[name], hide_index=True, width="stretch", column_config={
                "名次": st.column_config.NumberColumn(width=44),
                "鄰近詞": st.column_config.TextColumn(width=74),
                "cosine": st.column_config.NumberColumn(format="%.4f", width=56),
                "語境判讀": st.column_config.TextColumn(
                    "語境關係", width=121, help="顯示詞義、用法或共有的周邊詞。選取下方的近鄰可核對原文。"),
                "同篇共現（篇）": st.column_config.NumberColumn(
                    "共現篇數", width=70, help="一篇訓練新聞只要含有這兩個詞就計為 1 篇。兩詞可以出現在不同句子。"),
            })


def word_analysis(signature):
    heading("02 / WORDS", "詞向量分析", "找出出現在相似上下文的詞，再回到新聞內容理解它們的關係。")
    options = website_experiment_configs()
    _, reference_scores = read_weighting_reference()
    score_label = "單用詞向量的平均 nDCG@10"
    extremes = reference_model_extremes(options, ndcg10_by_model=reference_scores) if reference_scores else {}
    marks = {key: item[0] for key, item in extremes.items()}
    selected_runs = {}
    for col, method in zip(st.columns(2, gap="medium"), WORD_MODELS):
        configs = {name: models[method] for name, models in options.items() if method in models}
        selected_runs[method] = model_run_selectbox(col, method, configs, marks, f"word_{method}_run")
    if None in selected_runs.values():
        return

    st.caption("單用詞向量時，查詢與新聞各自將詞向量等權平均。"
               "選單的「最高」「最低」依固定 10 題關鍵字查詢的平均 nDCG@10 標註。")

    loaded_runs = {}
    models, metadata, model_signatures, selected_losses = {}, {}, {}, {}
    for method, name in selected_runs.items():
        try:
            if name not in loaded_runs:
                loaded_runs[name] = checked_experiment_resources(name)
            engine, manifest, _, path = loaded_runs[name]
            models[method] = engine.models[method]
            metadata[method] = manifest["models"][method]
            model_signatures[method] = sha256(path / "manifest.json")
            loss_digest = manifest["artifacts_sha256"][f"{method}/epoch_losses.csv"]
            score = reference_scores.get((name, method))
            if score is None:
                per_query = selected_unweighted_scores(model_signatures[method], method, query_bank(), engine)
                score = float(np.mean([row["ndcg10"] for row in per_query]))
            selected_losses[method] = (path, metadata[method]["config"], loss_digest, score)
        except (ValueError, OSError, KeyError, TypeError) as error:
            st.error(f"{WORD_MODELS[method]} 的 {name} 無法載入。{error}")
            return
    configs = {method: model["config"] for method, model in metadata.items()}
    st.markdown("**模型參數**")
    st.table(pd.DataFrame([
        {"模型": label, "向量維度": configs[name]["vector_size"], "窗口（window）": configs[name]["window"],
         "訓練輪數（epoch）": configs[name]["epochs"], "最低詞頻（min_count）": configs[name]["min_count"]}
        for name, label in WORD_MODELS.items()
    ]), hide_index=True, border="horizontal")
    st.subheader("訓練 Loss 比較")
    st.caption("比較最高、最低與目前選取的參數。每個點是該輪新增的訓練損失，兩種架構各用自己的縱軸刻度。")
    compare_extreme_losses(extremes, selected=selected_losses, score_label=score_label)

    default_word = "earnings"

    def fill_word():
        st.session_state["word_input"] = st.session_state["word_preset"]
    st.selectbox("範例詞", ANALYSIS_WORDS, index=ANALYSIS_WORDS.index(default_word), key="word_preset", on_change=fill_word)
    if "word_input" not in st.session_state:
        st.session_state["word_input"] = default_word
    with st.form("word_form"):
        st.text_input("分析一個英文詞", key="word_input")
        submitted = st.form_submit_button("查看 Top-5 近鄰", type="primary")
    if submitted:
        tokens = terms(st.session_state["word_input"])
        st.session_state["word_submitted"] = tokens[0] if len(tokens) == 1 else ""
    word = st.session_state.get("word_submitted", default_word)
    if not word:
        st.info("請輸入一個前處理後有效的英文詞。")
        return
    if terms(st.session_state["word_input"]) != [word]:
        st.caption(f"目前仍顯示已提交的 {word}；修改後請按「查看 Top-5 近鄰」。")
    report_signature = analysis_signature() if word in ANALYSIS_WORDS else ""
    entry = None
    if report_signature:
        payload, _ = report_resources(report_signature)
        entry = next((row for row in payload["words"] if row["word"] == word), None)
    points_by_model = {name: neighbor_points(word, name, model_signatures[name], model)
                       for name, model in models.items()}
    insights = neighbor_insights(word, entry, points_by_model, signature)
    widget_suffix = f"{word}_{':'.join(dict.fromkeys(selected_runs.values()))}"
    tables, all_rows = {}, []
    for name, points in points_by_model.items():
        rows = []
        for point in points:
            term = point["word"]
            insight = insights[term]
            evidence = insight["evidence"]
            cooccurrence = evidence["cooccurrence_documents"]
            rows.append({"名次": point["rank"], "鄰近詞": term, "cosine": point["cosine"],
                         "語境判讀": insight["relation"], "同篇共現（篇）": cooccurrence})
            witness_ids = dict.fromkeys(witness["id"] for witness in evidence["witnesses"])
            all_rows.append({"分析詞": word, "模型": name, "run": selected_runs[name], "名次": point["rank"], "word": term,
                             "cosine": point["cosine"], "relation": insight["relation"], "reading": insight["reading"],
                             "source": insight["source"], "cooccurrence_documents": cooccurrence,
                             "training_news_ids": "|".join(witness_ids), "unexpected": insight.get("unexpected")})
        tables[name] = pd.DataFrame(rows)
    word_map_panel(word, models, model_signatures, points_by_model, insights, configs, selected_runs, tables)
    st.info("向量相近可能反映共同詞形、相似新聞句式，甚至同一報表裡的反義詞。詞義要回到原文判讀，檢索效果另用測試新聞驗證。")
    if all_rows:
        st.download_button("下載這個詞的近鄰", csv_bytes(pd.DataFrame(all_rows)), f"neighbors-{word}.csv", "text/csv")
        word_questions(word, points_by_model, insights, configs, signature)
        word_interpretation(word, points_by_model, insights, widget_suffix, entry["evidence_terms"] if entry else ())
    with st.expander("閱讀訓練新聞中的使用情境"):
        for row in training_context(word, signature):
            st.markdown(f'**{row["id"]} · {row["title"]}**')
            st.write(row["excerpt"])
        st.caption("片段是使用情境範例，不是人工判定近鄰合理性的標準答案。")
    with st.expander("比較自訂詞對的 cosine"):
        a, b = st.columns(2)
        first = a.text_input("詞 A", value="profit", key="pair_a")
        second = b.text_input("詞 B", value="loss", key="pair_b")
        pair = [terms(first), terms(second)]
        if all(len(t) == 1 for t in pair):
            for name, (words, vectors) in models.items():
                positions = {term: i for i, term in enumerate(words)}
                if all(t[0] in positions for t in pair):
                    units = unit_rows(np.array([vectors[positions[t[0]]] for t in pair], dtype=np.float64))
                    st.write(f'{"CBOW" if name == "cbow" else "Skip-Gram"}：{pair[0][0]} ／ {pair[1][0]} cosine = {float(units[0] @ units[1]):.6f}')
                else:
                    st.caption(f"{name}：至少一詞沒有詞向量。")
        st.caption("例如 profit／loss 即使接近，也可能反映相似的財務句式，不能視為同義詞。")
    with st.expander("模型訓練資訊"):
        st.json(metadata, expanded=False)


def search_method_controls(options, reference_scores):
    extremes = reference_model_extremes(options, ndcg10_by_model=reference_scores) if reference_scores else {}
    marks = {key: item[0] for key, item in extremes.items()}
    selected_runs = {"tfidf": "正式模型"}
    method_labels = {"tfidf": METHODS["tfidf"]}
    a, b = st.columns([1, 2], gap="medium")
    with a.container(border=True, height="stretch", key="search_baseline"):
        st.caption("方法 A · 基準")
        st.markdown("### TF-IDF")
        st.write("用詞項權重表示查詢與新聞。")
    with b.container(border=True, height="stretch", key="search_embeddings"):
        st.caption("方法 B · 詞向量")
        st.markdown("### TF-IDF 加權詞向量")
        st.write("用 TF-IDF 權重平均詞向量，再比較查詢與新聞。")
        if marks:
            st.caption("預設選擇與「最高」「最低」標籤都依固定 10 題關鍵字查詢的單用詞向量平均 nDCG@10 決定。")
        for col, (method, label) in zip(st.columns(2), WORD_MODELS.items()):
            configs = {name: models[method] for name, models in options.items() if method in models}
            run = model_run_selectbox(col, method, configs, marks, f"search_{method}_run")
            if run is None:
                continue
            selected_runs[method] = run
            experiment_labels = distinct_run_labels({name: f"{label}-{model_setting_label(config)}"
                                                     for name, config in configs.items()})
            method_labels[method] = experiment_labels[run]
    return selected_runs, method_labels


def search_query_controls(bank, default):
    presets = {row["qid"]: row for row in bank}

    def choose(qid):
        st.session_state["query_preset"] = qid
        st.session_state["search_input"] = presets[qid]["text"]

    def fill_query():
        st.session_state["search_input"] = presets[st.session_state["query_preset"]]["text"]

    picker, alternatives = st.columns([2.5, 1.5], gap="medium")
    selected = picker.selectbox(
        "範例查詢", list(presets), index=list(presets).index(default), key="query_preset", on_change=fill_query,
        format_func=lambda key: f'{key} · {SEARCH_QUERY_TYPES[presets[key]["kind"]]} · '
                                f'{presets[key]["text"]}（{presets[key]["intent"]}）',
        help="模糊句避開該主題的原始關鍵字。混用保留原題部分關鍵字。",
    )
    picker.caption(f'{SEARCH_QUERY_TYPES[presets[selected]["kind"]]} · {presets[selected]["intent"]}')
    with alternatives:
        st.caption("同一題的三種問法")
        with st.container(horizontal=True, gap="small"):
            for query in bank:
                if query["base_qid"] == presets[selected]["base_qid"]:
                    st.button(SEARCH_QUERY_TYPES[query["kind"]], on_click=choose, args=(query["qid"],),
                              key=f'search_variant_{query["kind"]}', disabled=selected == query["qid"], help=query["text"])


@st.cache_data(show_spinner=False, max_entries=36)
def supplementary_scores(model_signature, method, queries, _engine):
    topics = {query["topic"] for query in queries}
    relevant = {topic: {doc["id"] for doc in _engine.documents if topic in doc["topics"]} for topic in topics}
    rows = {}
    for query in queries:
        ranking, _ = _engine.rank(query["text"], method)
        rows[query["qid"]] = metrics([item["id"] for item in ranking], relevant[query["topic"]])
    return rows


def explain_search_drops(scores, per_query, queries, models):
    baseline = scores["tfidf"]
    examples = []
    for method, score in scores.items():
        metric = next((key for key in ("ndcg10", "mrr", "p10") if score[key] is not None
                       and baseline[key] is not None and round(score[key] - baseline[key], 4) < 0), None)
        if metric is None:
            continue
        comparable = [query for query in queries if per_query[method][query["qid"]][metric] is not None]
        case = min(comparable, key=lambda query: per_query[method][query["qid"]][metric]
                   - per_query["tfidf"][query["qid"]][metric])
        examples.append((method, case))
    if not examples:
        return

    st.markdown("**加權後為什麼會掉分？**")
    st.caption("加權平均會把整篇新聞的詞向量合在一起，查詢中的精確詞項可能被其他相似用詞分散，"
               "讓語意接近但主題不同的新聞排得更前面。以下用實際排名核對掉分的情況。")
    for method, query in examples:
        before, after = (per_query[name][query["qid"]] for name in ("tfidf", method))
        first = [f'第 {round(1 / row["mrr"])} 名' if row["mrr"] else "無命中" for row in (before, after)]
        st.markdown(f'**{SEARCH_METHOD_NAMES[method]} · {query["qid"]}**　`{query["text"]}`')
        st.write(f'從 TF-IDF 換成加權模型後，首篇符合主題的新聞由 {first[0]} 變為 {first[1]}。'
                 f'前十名的主題命中數由 {round(before["p10"] * 10)} 篇變為 {round(after["p10"] * 10)} 篇，'
                 f'nDCG@10 由 {before["ndcg10"]:.4f} 變為 {after["ndcg10"]:.4f}。')
        engine, _ = models[method]
        ranking, status = engine.rank(query["text"], method)
        top = next((item for item in ranking[:10] if query["topic"] not in engine.by_id[item["id"]]["topics"]), None)
        if top:
            st.caption(f'加權模型第 {top["rank"]} 名（新聞 {top["id"]}）「{top["title"]}」'
                       f'未帶有這題的「{TOPICS[query["topic"]]}」主題標籤。')
        if status["embedding_weight_coverage"] < 1 - 1e-9:
            st.caption(f'這題有 {status["embedding_weight_coverage"]:.1%} 的 TF-IDF 權重可轉成詞向量。'
                       '部分用詞的權重因此沒有進入加權查詢。')


def format_search_score(value, baseline=None):
    if value is None:
        return "—"
    text = f"{value:.4f}"
    if baseline is None:
        return text
    delta = round(value - baseline, 4)
    return text + (f"（{delta:+.4f}）" if delta else "（持平）")


def paired_search_effects(bank, matched_query, models):
    queries = [query for query in bank if query["base_qid"] == matched_query["base_qid"]]
    original = queries[0]
    per_query = {method: supplementary_scores(signature, method, queries, engine)
                 for method, (engine, signature) in models.items()}
    table = []
    labels = {"ndcg10": "nDCG@10", "p10": "P@10", "mrr": "RR"}
    for query in queries:
        qid = query["qid"]
        for metric, label in labels.items():
            first_row = metric == "ndcg10"
            row = {"問法": SEARCH_QUERY_TYPES[query["kind"]] if first_row else "",
                   "英文查詢": query["text"] if first_row else "", "指標": label}
            baseline = per_query["tfidf"][qid][metric]
            for method, scores in per_query.items():
                row[SEARCH_METHOD_NAMES[method]] = format_search_score(scores[qid][metric],
                                                                       None if method == "tfidf" else baseline)
            table.append(row)
    st.caption(f'{original["qid"]} · {original["intent"]}。三種問法共用同一批測試新聞與 Reuters 主題判定。')
    with st.container(key="paired_query_table"):
        st.table(pd.DataFrame(table), hide_index=True, border="horizontal")
    st.caption("括號為相對同一列 TF-IDF 的差值。nDCG@10 看前十名排序，P@10 看前十名命中比例，"
               "RR 是第一篇符合主題的新聞名次倒數。")


def explain_retrieval_metrics():
    st.markdown("新聞含有這題預先指定的 Reuters 主題標籤，就算一篇命中。\n\n"
                "- **nDCG@10**　前十名每篇命中按名次得分。第 `i` 名得 `1 / log₂(i + 1)`，"
                "加總後除以理想排序的總分。理想排序把已知相關新聞排在最前面，最多計十篇。\n"
                "- **Precision@10（P@10）**　前十名的命中篇數除以 10。\n"
                "- **RR／MRR**　單題的 RR 是整份排名中第一篇命中新聞的名次倒數。"
                "整份排名皆未命中時為 0。多題的 RR 平均就是 MRR。")


def search_average_scores(kind, queries, evaluations, models):
    if kind == "keyword":
        per_query = {method: {row["qid"]: row for row in saved["per_query"] if row["method"] == method}
                     for method, saved in evaluations.items()}
    else:
        per_query = {method: supplementary_scores(signature, method, queries, engine)
                     for method, (engine, signature) in models.items()}
    included = [query["qid"] for query in queries if per_query["tfidf"][query["qid"]]["ndcg10"] is not None]
    scores = {method: {metric: float(np.mean([rows[qid][metric] for qid in included])) if included else None
                       for metric in METRICS} for method, rows in per_query.items()}
    return scores, per_query, included


def search_effects_overview(groups, evaluations, models):
    st.markdown("**核心比較 · 純 TF-IDF 與 TF-IDF 加權詞向量**")
    st.caption("各問法使用同一批測試新聞，依 Reuters 主題標籤判定命中。加權欄使用上方所選的模型。")
    table = []
    for kind, queries in groups.items():
        scores, _, included = search_average_scores(kind, queries, evaluations, models)
        scope = f"{SEARCH_QUERY_TYPES[kind]} {len(included)}／{len(queries)} 題"
        for metric, label in (("ndcg10", "nDCG@10"), ("p10", "P@10"), ("mrr", "MRR")):
            row = {"問法": scope if metric == "ndcg10" else "", "指標": label}
            baseline = scores["tfidf"][metric]
            for method, values in scores.items():
                row[SEARCH_METHOD_NAMES[method]] = format_search_score(values[metric], None if method == "tfidf" else baseline)
            table.append(row)
    st.table(pd.DataFrame(table), hide_index=True, border="horizontal")
    st.caption("括號為同一列相對純 TF-IDF 的差值。nDCG@10 看前十名排序，P@10 看命中比例，MRR 看首篇命中的名次。")
    st.write("詞向量可以讓用詞不同的相關新聞排前，也可能把語境相近的其他主題排前。"
             "加權平均把查詢與新聞各合成一個向量，詞序與否定關係容易被弱化。")


def explain_fuzzy_search(report):
    with st.expander("為什麼模糊句的 CBOW 和 Skip-Gram 表現差這麼多？"):
        # These fixed-query ranks and contributions were checked against this comparison's inputs.
        expected = "d352b8d6330d7a996e0fde8c1628b146e1bf84699f27780eff86de92415e81c1"
        try:
            verified = report is not None and sha256(COMPARISON / "manifest.json") == expected
        except OSError:
            verified = False
        if not verified:
            st.info("暫時無法核對這組模糊句案例的模型與評估版本。")
            return

        st.caption("固定案例 · CBOW v300-w15-e5 · Skip-Gram v150-w15-e15。上方成效表依目前所選模型更新。")
        st.write("純 TF-IDF 依字面重疊計分。兩種加權方法用相同的 TF-IDF 比重，詞向量的差異會把同一句查詢帶向不同主題。")
        st.caption("Q02-F 和 Q05-F 列出前十名符合 Reuters 主題標籤的篇數。")
        st.table(pd.DataFrame([
            ["Q02-F · 企業獲利", "4 / 10", "0 / 10", "8 / 10"],
            ["Q05-F · 原油生產", "5 / 10", "5 / 10", "10 / 10"],
            ["模糊句 10 題平均 nDCG@10", "0.5135", "0.4808", "0.7012"],
        ], columns=["案例／指標", "純 TF-IDF", "加權 CBOW", "加權 Skip-Gram"]), hide_index=True, border="horizontal")

        st.markdown(
            "**Q02-F · 最近三個月企業扣掉費用後賺了多少？**\n\n"
            "`How much money did businesses keep after expenses over the past three months?`\n\n"
            "純 TF-IDF 靠 `businesses` 的字面重疊，把一篇企業財報排第 2。"
            "加權 CBOW 裡，`money`、`keep` 等詞讓西班牙利率報導 #15552 排第 1，"
            "Medtronic 盈餘報導 #15213 排第 91。換成 Skip-Gram，`businesses` 和 `expenses` "
            "給盈餘報導的優勢足以反超，它升到第 1，利率報導降到第 13。"
        )
        st.markdown(
            "**Q05-F · 從地下井抽出了多少？**\n\n"
            "`How much is being pumped from underground wells?`\n\n"
            "純 TF-IDF 把 `wells` 對到銀行名稱 Wells Fargo，前兩名都是銀行新聞。"
            "加權 CBOW 排第 1 的是北海天然氣報導，這題以原油標籤計分。"
            "Skip-Gram 裡 `pumped` 的第一近鄰是 `oilfields`。整句加權後，"
            "用 `flow oil` 和 `exploratory well` 描述印尼近海油井的新聞 #20093 排第 1，"
            "這篇和查詢的用詞沒有字面重疊。"
        )
        st.info("**結論**　Q02-F 和 Q05-F 合計占 Skip-Gram 平均提升約一半。"
                "兩種加權方法能否把相關新聞排前，取決於所用詞向量。"
                "同一批模糊句改用 CBOW `v150-w15-e15`，加權 nDCG@10 為 **0.5622**，高於純 TF-IDF 的 **0.5135**。")


def explain_search_average(kind, report, baseline_by_query):
    label = SEARCH_QUERY_TYPES[kind]
    with st.expander(f"為什麼{label}查詢加權後會變好或變差？"):
        try:
            examples = read_json(ROOT / "resources" / "search_average_examples.json")
            if report is None or sha256(COMPARISON / "manifest.json") != examples["comparison_manifest_sha256"]:
                raise ValueError("The examples belong to another comparison")
            group = examples["groups"][kind]
            queries = {query["qid"]: query for query in report["queries"] if query["kind"] == kind}
            runs = {(row["method"], row["run"]): row for row in report["runs"]}
            cbow, skipgram = (runs[method, examples["runs"][method]] for method in ("cbow", "skipgram"))
        except (OSError, ValueError, KeyError, TypeError):
            st.info(f"暫時無法核對這組{label}案例的模型與評估版本。")
            return

        st.caption(f"固定案例 · CBOW {model_setting_label(cbow['config'])} · "
                   f"Skip-Gram {model_setting_label(skipgram['config'])}。上方成效表依目前所選模型更新。")
        st.caption("每題顯示前十名命中數與 nDCG@10。命中依 Reuters 主題標籤判定，名次來自完整測試集。")
        scores = [
            baseline_by_query,
            {qid: cbow["per_query"][qid]["weighted"] for qid in queries},
            {qid: skipgram["per_query"][qid]["weighted"] for qid in queries},
        ]
        methods = ("tfidf", "cbow", "skipgram")
        columns = ["純 TF-IDF", SEARCH_METHOD_NAMES["cbow"], SEARCH_METHOD_NAMES["skipgram"]]
        summary_rows = []
        for case in group["cases"]:
            qid = case["qid"]
            values = [method_scores[qid] for method_scores in scores]
            summary_rows.append([f"{qid} · {TOPICS[queries[qid]['topic']]}",
                                 *(f"{round(row['p10'] * 10)} / 10 · {row['ndcg10']:.4f}" for row in values)])
        summary_rows.append([f"{label} {len(queries)} 題平均 nDCG@10",
                             *(f"{np.mean([rows[qid]['ndcg10'] for qid in queries]):.4f}" for rows in scores)])
        st.table(pd.DataFrame(summary_rows, columns=["案例／指標", *columns]), hide_index=True, border="horizontal")

        for case in group["cases"]:
            query = queries[case["qid"]]
            st.markdown(f"#### {query['qid']} · {query['intent']}")
            st.markdown(f"`{query['text']}`")
            article_rows = [
                [f"#{article['id']} · {article['title']}", "、".join(article["topics"]) or "無標籤",
                 *(f"第 {article['ranks'][method]:,} 名" for method in methods)]
                for article in case["articles"]
            ]
            st.table(pd.DataFrame(article_rows, columns=["新聞", "Reuters 主題", *columns]),
                     hide_index=True, border="horizontal")
            st.markdown(case["reading"])
        st.info(f"**結論**　{group['conclusion']}")


def search_effects(evaluations, bank, matched_query, models, results, relevant, report):
    with st.container(border=True, key="search_effects"):
        st.markdown("### 檢索有改善嗎？")
        groups = {kind: [query for query in bank if query["kind"] == kind] for kind in SEARCH_QUERY_TYPES}
        scopes = {f"{label} {len(groups[kind])} 題平均": kind for kind, label in SEARCH_QUERY_TYPES.items()}
        paired, overview = "同題三種問法", "三種問法總覽"
        scope = st.radio("成效比較範圍", ("目前查詢", overview, paired, *scopes), horizontal=True,
                         label_visibility="collapsed", key="search_effect_scope")
        if scope == overview:
            search_effects_overview(groups, evaluations, models)
            return
        kind = scopes.get(scope)
        average = kind is not None
        if not average and matched_query is None:
            st.info("目前查詢沒有預先指定的相關主題。可切到任一查詢類型的平均，比較所選模型的成效。")
            return
        if scope == paired:
            paired_search_effects(bank, matched_query, models)
            return
        if average:
            queries = groups[kind]
            scores, per_query, included = search_average_scores(kind, queries, evaluations, models)
            st.caption(f"{SEARCH_QUERY_TYPES[kind]}測試納入 {len(included)}／{len(queries)} 題。三種問法共用相同測試新聞及主題判定。")
        else:
            queries = [matched_query]
            qid = matched_query["qid"]
            scores = {method: metrics([item["id"] for item in ranking], relevant)
                      for method, (ranking, _) in results.items()}
            per_query = {method: {qid: score} for method, score in scores.items()}
            st.caption(f"{qid} · {SEARCH_QUERY_TYPES[matched_query['kind']]} · {matched_query['intent']}。"
                       f"測試集中有 {len(relevant)} 篇符合 Reuters 主題 {matched_query['topic']}。")
        baseline = scores["tfidf"]
        if baseline["ndcg10"] is None:
            st.info("這個範圍沒有已知相關新聞可供評分。")
            return

        labels = {"ndcg10": "nDCG@10 · 前十名排序", "p10": "P@10 · 前十名命中率",
                  "mrr": "MRR · 首篇相關排名" if average else "RR · 首篇相關排名"}
        table = []
        outcomes = []
        for metric, label in labels.items():
            row = {"指標": label}
            for method, score in scores.items():
                name, value = SEARCH_METHOD_NAMES[method], score[metric]
                row[name] = format_search_score(value, None if method == "tfidf" else baseline[metric])
                if metric == "ndcg10" and method != "tfidf" and value is not None:
                    delta = round(value - baseline[metric], 4)
                    change = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
                    outcomes.append(f"**{name}** {change}")
            table.append(row)
        if outcomes:
            st.markdown(f"以 nDCG@10 衡量前十名排序，相對於 TF-IDF，{'、'.join(outcomes)}。")
        st.table(pd.DataFrame(table), hide_index=True, border="horizontal")
        st.caption("括號為相對 TF-IDF 的差值。相關新聞以 Reuters 主題標籤判定。")
        if kind == "fuzzy":
            explain_fuzzy_search(report)
        elif average:
            explain_search_average(kind, report, per_query["tfidf"])
        explain_search_drops(scores, per_query, queries, models)
        with st.expander("指標怎麼看？"):
            explain_retrieval_metrics()
            st.write("主題標籤用來衡量新聞是否符合查詢的大方向。加權平均會把一段文字合成單一向量，"
                     "因此語序和否定等差異容易被壓縮。")


def rank_weighting_rows(rows, metric):
    values = pd.DataFrame(rows, columns=[f"unweighted_{metric}", f"weighted_{metric}"])
    ranks = values.round(12).rank(ascending=False, method="min").astype(int)
    ranked = [{**row, "unweighted_rank": before, "weighted_rank": after}
              for row, (before, after) in zip(rows, ranks.itertuples(index=False, name=None), strict=True)]
    return sorted(ranked, key=lambda row: (row["weighted_rank"], row["unweighted_rank"], row["run"]))


def embedding_weighting_table(rows, metric, label):
    before_key, after_key = f"unweighted_{metric}", f"weighted_{metric}"
    return pd.DataFrame([
        {"參數": model_setting_label(row["config"]),
         f"單用 {label}": f'{row[before_key]:.4f}（第 {row["unweighted_rank"]} 名）',
         f"加權 {label}": f'{row[after_key]:.4f}（第 {row["weighted_rank"]} 名）',
         "加權差值": f'{row[after_key] - row[before_key]:+.4f}'}
        for row in rows
    ])


def weighting_row_for_queries(row, queries):
    pairs = {query["qid"]: row["per_query"][query["qid"]] for query in queries}
    scores = {f"{mode}_{metric}": float(np.mean([pair[mode][metric] for pair in pairs.values()]))
              for mode in ("unweighted", "weighted") for metric in METRICS}
    return {**row, **scores, "per_query": pairs}


@st.cache_data(show_spinner=False, max_entries=12)
def selected_unweighted_scores(model_signature, method, queries, _engine):
    count_space = replace(_engine.space)
    count_space.idf.fill(1.)
    documents = count_space.transform([terms(document_text(doc)) for doc in _engine.documents])
    tokens = [terms(query["text"]) for query in queries]
    relevant = [{doc["id"] for doc in _engine.documents if query["topic"] in doc["topics"]} for query in queries]
    return unweighted_scores(_engine, method, documents, count_space.transform(tokens), tokens, relevant)


def custom_weighting_row(config, method, queries, model, evaluation):
    engine, signature = model
    unweighted = selected_unweighted_scores(signature, method, queries, engine)
    weighted = {row["qid"]: row for row in evaluation["per_query"] if row["method"] == method}
    pairs = {}
    for query, before in zip(queries, unweighted, strict=True):
        qid = query["qid"]
        after = weighted.get(qid)
        if after is None:
            relevant = {doc["id"] for doc in engine.documents if query["topic"] in doc["topics"]}
            after = weighted_query_score(engine, method, query["text"], relevant)
        if after["ndcg10"] is None or after["relevant_total"] != before["relevant_total"]:
            raise ValueError("查詢的相關新聞篇數與參考實驗不符。")
        pairs[qid] = {"unweighted": before, "weighted": after}
    return weighting_row_for_queries({"config": config, "per_query": pairs}, queries)


def weighting_score_card(rows, current, metric, score_label, metric_labels):
    detail_labels = {**metric_labels, "p10": "P@10"}
    selection = '<p class="weighting-selection">從上方選取這個架構的模型，即可查看分數。</p>'
    if current is not None:
        params = escape(model_setting_label(current["config"]))
        delta = current[f"weighted_{metric}"] - current[f"unweighted_{metric}"]
        selection = (f'<div class="weighting-selection"><span class="weighting-badge weighting-current">目前選取</span>'
                     f'<code>{params}</code><span class="weighting-delta">加權差值 {delta:+.4f}</span></div>')

    panels = []
    for mode, title in (("unweighted", "單用詞向量"), ("weighted", "TF-IDF 加權")):
        score_key, rank_key = f"{mode}_{metric}", f"{mode}_rank"
        score, rank_text, rank_note = "—", "—", ""
        detail = ""
        if current is not None:
            score = f'{current[score_key]:.4f}'
            rank = current.get(rank_key)
            rank_text = "參考組外"
            if rank is not None:
                tied_count = sum(row[rank_key] == rank for row in rows)
                mark = "並列" if tied_count > 1 else ""
                rank_text = f"{mark}第 {rank} 名"
                rank_note = f"共 {len(rows)} 組"
                if tied_count == len(rows):
                    rank_text = "全部同分"
                elif tied_count > 1:
                    rank_note += f" · {tied_count} 組同分"
            other_scores = "".join(f'<span>{label} <b>{current[f"{mode}_{key}"]:.4f}</b></span>'
                                   for key, label in detail_labels.items() if key != metric)
            detail = f'<div class="weighting-metrics">{other_scores}</div>'
        extremes = []
        worst_rank = max(row[rank_key] for row in rows)
        bounds = (("全部同分", 1, "high"),) if worst_rank == 1 else (("最高", 1, "high"), ("最低", worst_rank, "low"))
        for name, position, color in bounds:
            group = [row for row in rows if row[rank_key] == position]
            badge = f"並列{name}" if len(group) > 1 and worst_rank > 1 else name
            sample = next((row for row in group if row is current), group[0])
            setting = escape(model_setting_label(sample["config"]))
            tied = f'<small class="weighting-tie">另 {len(group) - 1} 組</small>' if len(group) > 1 else ""
            extremes.append(f'<div class="weighting-extreme"><span class="weighting-badge weighting-{color}">{badge}</span>'
                            f'<strong>{sample[score_key]:.4f}</strong><code>{setting}</code>{tied}</div>')
        panels.append(f'<div class="weighting-score weighting-{mode}"><div class="weighting-mode">{title}</div>'
                      f'<div class="weighting-scoreline"><span>{score_label}</span><strong>{score}</strong></div>'
                      f'<div class="weighting-rank">參數排名 <strong>{rank_text}</strong><small>{rank_note}</small></div>'
                      f'{detail}{"".join(extremes)}</div>')
    st.html(f'{selection}<div class="weighting-pair">{"".join(panels)}</div>')
    if current is not None:
        queries = current["per_query"].values()
        missing = sorted({word for pair in queries for word in pair["unweighted"]["missing_terms"]})
        first = next(iter(queries))
        before, after = first["unweighted"]["valid_documents"], first["weighted"]["valid_documents"]
        st.caption(f'可排名新聞 {before:,} → {after:,} 篇（單用 → 加權）。缺向量詞 {"、".join(missing) or "無"}。')
        valid_before, valid_after = (sum(pair[mode]["valid_query"] for pair in queries) for mode in ("unweighted", "weighted"))
        if min(valid_before, valid_after) < len(queries):
            st.caption(f"有效查詢 {valid_before}／{len(queries)} → {valid_after}／{len(queries)} 題（單用 → 加權）。")


def embedding_weighting_example(queries, standalone, weighted, metric):
    def gain(row, qid):
        pair = row["per_query"][qid]
        return pair["weighted"][metric] - pair["unweighted"][metric]

    same_run = standalone["run"] == weighted["run"]
    sample = max(queries, key=lambda q: abs(gain(weighted, q["qid"])) if same_run
                 else gain(weighted, q["qid"]) - gain(standalone, q["qid"]))
    qid = sample["qid"]
    label = WORD_MODELS[weighted["method"]]
    example_metric = "RR" if metric == "mrr" else METRICS[metric]
    with st.expander(f"{label} 的查詢案例 · {qid} {sample['intent']}"):
        st.markdown(f'`{sample["text"]}`')
        for row in ([weighted] if same_run else [weighted, standalone]):
            pair = row["per_query"][qid]
            before, after = pair["unweighted"], pair["weighted"]
            first = [f'第 {round(1 / item["mrr"])} 名' if item["mrr"] else "無命中" for item in (before, after)]
            st.markdown(f'**{model_setting_label(row["config"])}**　{example_metric} **{before[metric]:.4f} → {after[metric]:.4f}**。'
                        f'前十名命中 {round(before["p10"] * 10)} → {round(after["p10"] * 10)} 篇，'
                        f'首篇命中由 {first[0]} 變為 {first[1]}。')
            missing = "、".join(before["missing_terms"]) or "無"
            st.caption(f'單用 → 加權可排名的新聞 {before["valid_documents"]:,} → {after["valid_documents"]:,} 篇。'
                       f'缺向量的查詢詞 {missing}。加權涵蓋 {after["embedding_weight_coverage"]:.1%}。')
        weights = sorted(sample["tfidf_weights"].items(), key=lambda item: -item[1])
        if weights:
            st.caption("本題查詢詞的 TF-IDF 權重為 " + "、".join(f'`{word}` {score:.3f}' for word, score in weights) + "。")
            st.write(f'`{weights[0][0]}` 在查詢中權重最大，可能讓排序更偏向這個詞在各模型中學到的語境。'
                     '新聞端的詞也同時重新加權，這些名次變化反映兩邊合起來的結果。')


def explain_weighting_winners(runs):
    with st.expander("為什麼單用最高，加權後可能換一組？"):
        st.markdown(
            "**兩欄都在比新聞排序，差在怎麼把詞合成向量。** 同一列使用同一組詞向量。"
            "單用時，查詢和每篇新聞裡，有向量的詞每出現一次就算一份，再取平均。"
            "加權時，TF-IDF 較高的詞在平均裡占得更多。\n\n"
            "**名次為什麼會換？** 不同參數學到的詞向量不同。加重某個詞後，有的模型會把符合主題的新聞往前排，"
            "有的模型則可能把其他主題的新聞帶進前十名。各組分數的變化不同，原本的第一名就可能被超過。"
        )
        st.caption(f"兩欄用同一批測試新聞與關鍵字 10 題的平均 nDCG@10。"
                   f"每個架構各比較 {len(REFERENCE_RUNS)} 組。前十名命中的相關新聞越多、排得越前，分數越高。")
        tables = []
        changed_winners = []
        skipgram_example = None
        for method, label in WORD_MODELS.items():
            rows = rank_weighting_rows([row for row in runs if row["method"] == method], "ndcg10")
            standalone = next(row for row in rows if row["unweighted_rank"] == 1)
            weighted = rows[0]
            if standalone["run"] == weighted["run"]:
                winners, roles = [standalone], ["兩欄最高"]
            else:
                winners, roles = [standalone, weighted], ["單用最高", "加權最高"]
                if standalone["weighted_rank"] != 1:
                    changed_winners.append(label)
                if method == "skipgram":
                    skipgram_example = standalone, weighted
            frame = embedding_weighting_table(winners, "ndcg10", "nDCG@10").drop(columns="加權差值")
            frame.insert(0, "架構", label)
            frame.insert(1, "最高欄位", roles)
            tables.append(frame)
        st.table(pd.concat(tables, ignore_index=True), hide_index=True, border="horizontal")
        if skipgram_example is not None:
            standalone, weighted = skipgram_example
            st.markdown(
                f"**看 Skip-Gram 這兩列。** 單用第一名 `{model_setting_label(standalone['config'])}` "
                f"的分數在加權後從 **{standalone['unweighted_ndcg10']:.4f}** 變成 **{standalone['weighted_ndcg10']:.4f}**。"
                f"另一組 `{model_setting_label(weighted['config'])}` 從 **{weighted['unweighted_ndcg10']:.4f}** "
                f"變成 **{weighted['weighted_ndcg10']:.4f}**，因此在加權欄排第一。"
            )
        result = (
            f"這 10 題中，{' 和 '.join(changed_winners)} 的單用第一名在加權後被其他組超過。"
            if changed_winners else "這 10 題中，各架構的單用第一名在加權後仍是最高。"
        )
        st.info(f"**結論**　{result} TF-IDF 重新分配詞的比重，各組把相關新聞往前排的效果也會改變。"
                "比較加權檢索的參數時，要看同一批查詢的加權 nDCG@10。")


def explain_palm_oil_ranking():
    st.markdown("#### 棕櫚油為什麼排在原油前面？")
    translations = {"oil": "油", "output": "產量", "prices": "價格"}
    try:
        # This fixed case was computed offline with the verified retrieval engine.
        example = read_json(ROOT / "resources" / "palm_oil_ranking_example.json")
        _, _, manifest, evaluation = completed_manifests(example["run"])
        if example["model_manifest_sha256"] != evaluation["model_manifest_sha256"]:
            raise ValueError("The ranking example belongs to a different model")
        config = manifest["models"]["skipgram"]["config"]
        query = example["query"]
        closest = example["nearest_to_oil"]
        crude, palm = (example["documents"][doc_id] for doc_id in ("18678", "14892"))
        tokens = terms(query)
        parts = np.array([[row["contributions"][token] for row in (crude, palm)] for token in tokens])
        if not np.allclose(parts.sum(axis=0), [crude["score"], palm["score"]], rtol=0, atol=1e-9):
            raise ValueError("Query term contributions do not match the retrieval scores")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        st.info("暫時無法核對這個實例的模型與搜尋分數。")
        return

    st.caption(f"固定使用 Skip-Gram · {model_setting_label(config)}（{model_setting_description(config)}）。")
    st.markdown(f"查詢 `{query}`（油的產量與價格）。這組模型中，`oil` 的第 1 近鄰是 `{closest}`。"
                "此例想找 Reuters 標為 `crude` 的原油新聞。")
    st.caption("原油 #18678 · 帝國石油調漲原油牌價。棕櫚油 #14892 · 馬來西亞棕櫚油產量恐低於目標。")
    score_gap = palm["score"] - crude["score"]
    rows = [[f"{token}（{translations[token]}）", f"{left:+.4f}", f"{right:+.4f}", f"{right - left:+.4f}"]
            for token, (left, right) in zip(tokens, parts, strict=True)]
    rows.extend([
        ["總分（cosine）", f"{crude['score']:.4f}", f"{palm['score']:.4f}", f"{score_gap:+.4f}"],
        ["搜尋名次", f"第 {crude['rank']} 名", f"第 {palm['rank']} 名", "—"],
    ])
    st.table(pd.DataFrame(rows, columns=["查詢詞／結果", "原油 #18678", "棕櫚油 #14892", "棕櫚油減原油"]),
             hide_index=True, border="horizontal")
    st.caption(f"前三列相加就是 cosine 總分。名次來自同一查詢在 {manifest['test_documents']:,} 篇測試新聞中的完整搜尋。")
    oil_lead = parts[0, 0] - parts[0, 1]
    palm_lead = parts[1:, 1].sum() - parts[1:, 0].sum()
    st.markdown(
        f"**原油在 `oil` 這項領先 {oil_lead:.4f}。** 棕櫚油那篇也在談產量和價格，"
        f"在 `output` 和 `prices` 合計多拿 **{palm_lead:.4f}**，所以總分反超 **{score_gap:.4f}**。"
        "模型在 `crude oil` 和 `palm oil` 中共用 `oil` 的向量，搜尋時再把各詞加權平均。"
    )


def explain_word_neighbors_and_retrieval(report):
    with st.expander("為什麼近鄰詞更接近，加權檢索未必更好？"):
        explain_palm_oil_ranking()
        st.divider()
        st.markdown("#### Q05 · 比較兩組模型的近鄰與檢索")
        st.write("用 Q05「crude oil production」（原油生產）來看。兩組 Skip-Gram 都是 100 維、窗口 15，"
                 "訓練輪數分別是 5 輪和 15 輪。")
        skipgram_runs = {row["run"]: row for row in report["runs"] if row["method"] == "skipgram"}
        query = next(row for row in report["queries"] if row["qid"] == "Q05")
        weights = query["tfidf_weights"]
        total_weight = sum(weights.values())
        weight_text = "、".join(f"`{term}` **{weights[term] / total_weight:.1%}**" for term in terms(query["text"]))
        rows = []
        try:
            for name in ("grid-d100-w15-e5", "grid-d100-w15-e15"):
                result = skipgram_runs[name]
                run, _, manifest, _ = completed_manifests(name)
                filename = "word_neighbors.csv"
                raw = checked_bytes(run, filename, manifest["artifacts_sha256"][filename])
                frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
                pair = frame.loc[
                    (frame["model"] == "skipgram") & (frame["anchor"] == "oil") & (frame["word"] == "petroleum")
                ].iloc[0]
                score = result["per_query"]["Q05"]["weighted"]
                rows.append({
                    "Skip-Gram 參數": model_setting_label(result["config"]),
                    "petroleum 的近鄰名次": f"第 {pair['rank']} 名",
                    "oil 與 petroleum 的 cosine": f"{pair['cosine']:.4f}",
                    "Q05 前十名命中": f"{score['p10'] * 10:.0f} / 10",
                    "Q05 加權 nDCG@10": f"{score['ndcg10']:.4f}",
                    "10 題平均加權 nDCG@10": f"{result['weighted_ndcg10']:.4f}",
                })
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            st.info("這個實例的近鄰與檢索分數暫時無法核對。")
            return

        st.caption("近鄰以 oil 為中心。Q05 欄只評這一題，最後一欄才是關鍵字 10 題平均。")
        st.table(pd.DataFrame(rows), hide_index=True, border="horizontal")
        five, fifteen = rows
        st.markdown(
            f"**1. 看兩個詞。** 5 輪的 `oil` 和 `petroleum` 比較接近，cosine 是 **{five['oil 與 petroleum 的 cosine']}**，"
            f"略高於 15 輪的 **{fifteen['oil 與 petroleum 的 cosine']}**。這個數字只看這一對詞。\n\n"
            f"**2. 看整句查詢和整篇新聞。** 合成查詢向量時，三個詞的 TF-IDF 比重是 {weight_text}。"
            "`crude` 和 `production` 各比 `oil` 占得多。每篇新聞也會把自己的詞加權平均，"
            "最後依整句查詢和整篇新聞的 cosine 排名。\n\n"
            "**3. 看前十名的實際結果。** 新聞 #20406 標題有 `petroleum`，但 Reuters 標的是 `fuel`（燃料）。"
            "這題以 `crude`（原油）為命中條件，所以這篇不算命中。"
            f"5 輪把它排在第 7 名，前十名命中 **{five['Q05 前十名命中']}**。"
            f"15 輪把它排在第 28 名，前十名命中 **{fifteen['Q05 前十名命中']}**，"
            f"所以 Q05 的 nDCG@10 從 **{five['Q05 加權 nDCG@10']}** 提到 **{fifteen['Q05 加權 nDCG@10']}**。"
        )
        st.info("**結論**　近鄰的 cosine 描述兩個詞的關係。加權檢索還會合併整句查詢與整篇新聞的其他用詞，"
                "所以棕櫚油能靠產量和價格排前，Q05 也由近鄰略遠的 15 輪模型命中更多原油新聞。"
                "檢索有沒有改善，要用同一查詢比較相關新聞的命中數和位置。")


def embedding_weighting_panel(report, selected_runs, options, models, evaluations, matched_query):
    st.subheader("詞向量參數表現")
    if report is None:
        st.info("完成並驗證參考組的詞向量加權比較後，這裡會顯示兩種排名。")
        return

    groups = {kind: [query for query in report["queries"] if query["kind"] == kind] for kind in SEARCH_QUERY_TYPES}
    scopes = {f"{label} {len(groups[kind])} 題平均": kind for kind, label in SEARCH_QUERY_TYPES.items()}
    scope = st.radio("詞向量參數比較範圍", ("目前查詢", *scopes), horizontal=True, key="embedding_weighting_scope")
    single = scope == "目前查詢"
    metric_labels = {"ndcg10": METRICS["ndcg10"], "p10": METRICS["p10"], "mrr": "RR" if single else METRICS["mrr"]}
    metric = st.radio("參數排名指標", tuple(metric_labels), horizontal=True, key="embedding_weighting_metric",
                      format_func=lambda key: "RR／MRR" if key == "mrr" else metric_labels[key],
                      help="nDCG@10 看前十名的命中與位置。Precision@10 看前十名命中比例。RR／MRR 看首篇命中的名次，單題用 RR，多題用 MRR。")
    metric_label = metric_labels[metric]
    score_label = f"平均 {metric_label}" if not single and metric != "mrr" else metric_label
    if single and matched_query is None:
        st.info("這個查詢還沒有預先指定的 Reuters 主題。可提交一個範例查詢，或改看任一題型的平均分數。")
        with st.expander("指標怎麼算？"):
            explain_retrieval_metrics()
        explain_word_neighbors_and_retrieval(report)
        return

    if single:
        query = next(row for row in report["queries"] if row["qid"] == matched_query["qid"])
        queries = [query]
        relevant = report["runs"][0]["per_query"][query["qid"]]["unweighted"]["relevant_total"]
        st.markdown(f'**{query["qid"]} · {SEARCH_QUERY_TYPES[query["kind"]]}**　`{query["text"]}`（{query["intent"]}）')
        st.caption(f'{report["test_documents"]:,} 篇測試新聞中，有 {relevant:,} 篇符合「{TOPICS[query["topic"]]}」主題。')
    else:
        kind = scopes[scope]
        queries = groups[kind]
        st.caption(f'使用 {queries[0]["qid"]} 至 {queries[-1]["qid"]} 的「{SEARCH_QUERY_TYPES[kind]}」問法，'
                   f'共 {len(queries)} 題。每題在同一批 {report["test_documents"]:,} 篇測試新聞上評分。')
        with st.expander(f"查看這 {len(queries)} 題的查詢內容"):
            st.table(pd.DataFrame([{"題號": query["qid"], "英文查詢": query["text"], "中文意思": query["intent"]}
                                   for query in queries]), hide_index=True, border="horizontal")
    st.caption(f"每欄依 {score_label} 在同一架構的 {len(REFERENCE_RUNS)} 組參考參數中排名。同分共用名次。")
    st.write("單用把詞向量等權平均，加權按 TF-IDF 分配比重。")
    rankings, examples = {}, []
    for col, (method, label) in zip(st.columns(2, gap="medium"), WORD_MODELS.items()):
        rows = [weighting_row_for_queries(row, queries) for row in report["runs"] if row["method"] == method]
        rows = rank_weighting_rows(rows, metric)
        rankings[method] = rows
        examples.append((next(row for row in rows if row["unweighted_rank"] == 1), rows[0]))
        run = selected_runs.get(method)
        current = next((row for row in rows if row["run"] == run), None)
        with col.container(border=True, height="stretch"):
            st.markdown(f"#### {label}")
            if run is not None and current is None:
                try:
                    current = custom_weighting_row(options[run][method], method, queries, models[method], evaluations[method])
                except (ValueError, KeyError, TypeError):
                    st.warning("目前選取的模型分數無法核對。")
            weighting_score_card(rows, current, metric, score_label, metric_labels)

    st.caption(f"加權差值為 {score_label} 的加權分數減單用分數。參考組外的模型也用相同查詢評分。v 是維度，w 是窗口，e 是訓練輪數。")
    if single:
        st.caption("同分並列。卡片列出其中一組及其餘組數，完整名單可展開參數表查看。")
    with st.expander("指標怎麼算？"):
        explain_retrieval_metrics()
    explain_weighting_winners(report["runs"])
    explain_word_neighbors_and_retrieval(report)
    for standalone, weighted in examples:
        embedding_weighting_example(queries, standalone, weighted, metric)
    with st.expander("查看所有參數組合的兩種名次"):
        for tab, rows in zip(st.tabs(list(WORD_MODELS.values())), rankings.values()):
            tab.dataframe(embedding_weighting_table(rows, metric, metric_label), hide_index=True, width="stretch")


def search_final_conclusion(bank, evaluations, models, selected_runs, method_labels, report):
    with st.container(border=True, key="search_final_conclusion"):
        st.markdown("### TF-IDF 和 TF-IDF 加權詞向量，哪個好？")
        st.caption("目前比較 · " + " · ".join(method_labels.values()))
        st.caption("各問法用同一批測試新聞，依 Reuters 主題標籤評分。表中是平均 nDCG@10，括號是相對純 TF-IDF 的差值。")

        # Use the verified comparison for reference runs and the regular scorer for other selections.
        references = {(row["method"], row["run"]): row["per_query"] for row in report["runs"]} if report else {}
        saved = {method: references[method, run] for method, run in selected_runs.items()
                 if (method, run) in references}
        live_models = {method: model for method, model in models.items() if method not in saved}
        live_evaluations = {method: evaluations[method] for method in live_models}
        names = {**SEARCH_METHOD_NAMES, "tfidf": "純 TF-IDF"}
        common_winners = {names[method] for method in models}
        table = []
        for kind, label in SEARCH_QUERY_TYPES.items():
            queries = [query for query in bank if query["kind"] == kind]
            scores, _, included = search_average_scores(kind, queries, live_evaluations, live_models)
            values = {method: row["ndcg10"] for method, row in scores.items()}
            values.update({method: float(np.mean([rows[qid]["weighted"]["ndcg10"] for qid in included]))
                           if included else None for method, rows in saved.items()})
            rounded = {method: round(value, 4) for method, value in values.items() if value is not None}
            top = max(rounded.values(), default=None)
            winners = [names[method] for method, value in rounded.items() if value == top]
            common_winners.intersection_update(winners)
            table.append({
                "問法": f"{label} {len(included)}／{len(queries)} 題",
                **{names[method]: format_search_score(values[method], None if method == "tfidf" else values["tfidf"])
                   for method in models},
                "此問法最高": ("並列 " if len(winners) > 1 else "") + ("、".join(winners) or "—"),
            })
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

        st.markdown(
            "- **查詢用字明確，新聞也使用同樣的詞。** 純 TF-IDF 可直接利用字面重疊。"
            "例如查詢 `company acquisition merger`，新聞裡的 `merger` 已經是明確的併購線索。\n"
            "- **同一主題有不同說法。** 加權詞向量可以補上字面配不到的線索。"
            "例如油井查詢用 `pumped` 和 `wells`，原油新聞可能寫 `flow oil` 和 `exploratory well`。\n"
            "- **關鍵字加上描述。** 描述指向同一主題時，能幫忙找出用詞不同的新聞。"
            "描述落在相近但不同的主題時，也會把它們往前推，例如 `deficit` 同時出現在貿易逆差和經常帳赤字的報導。"
        )
        result = (f"目前的 **{next(iter(common_winners))}** 在三種問法的平均 nDCG@10 都達到最高。"
                  if len(common_winners) == 1 else "選擇方法時，對照你常用的問法和該列的最高分。")
        st.info(f"**結論**　{result} 加權詞向量有沒有幫助，要看相關新聞在前十名命中了幾篇、排在多前面。"
                "單詞加權平均容易弱化詞序與否定，因此還要用實際新聞名次核對。")


def news_search(engine, signature):
    heading("03 / RETRIEVAL", "新聞搜尋與比較", "比較相同查詢的新聞排名，再用固定實驗檢查檢索成效。")
    options = website_experiment_configs()
    report, reference_scores = read_weighting_reference()
    selected_runs, method_labels = search_method_controls(options, reference_scores)
    bank = search_query_bank()
    default = bank[0]
    search_query_controls(bank, default["qid"])
    if "search_input" not in st.session_state:
        st.session_state["search_input"] = default["text"]
    with st.form("search_form"):
        st.text_input("英文查詢", key="search_input", placeholder=f'例如 {default["text"]}')
        submitted = st.form_submit_button("搜尋並比較", type="primary")
    if submitted:
        st.session_state["submitted_query"] = st.session_state["search_input"]
    query = st.session_state.get("submitted_query", default["text"])
    if query != st.session_state["search_input"]:
        st.caption(f"目前結果屬於「{query}」。修改後請按「搜尋並比較」。")
    engines = {selected_runs["tfidf"]: engine}
    signatures = {selected_runs["tfidf"]: signature}
    saved_runs = {}
    for method, run in selected_runs.items():
        if run in engines:
            continue
        try:
            selected_engine, _, evaluation, path = checked_experiment_resources(run)
            if selected_engine.ids != engine.ids:
                raise ValueError("測試新聞清單與正式模型不符。")
        except (ValueError, OSError, KeyError, TypeError) as error:
            st.error(f"{method_labels[method]} 無法載入。{error}")
            return
        engines[run] = selected_engine
        signatures[run] = sha256(path / "manifest.json")
        saved_runs[run] = evaluation
    results = {method: engines[run].rank(query, method) for method, run in selected_runs.items()}
    query_key = " ".join(normalize(query).split())
    matched = next((item for item in bank if " ".join(normalize(item["text"]).split()) == query_key), None)
    relevant = {doc["id"] for doc in engine.documents if matched and matched["topic"] in doc["topics"]}
    st.subheader(f"搜尋結果：{query or '（空白查詢）'}")
    st.caption(f"各方法搜尋同一批 {len(engine.ids):,} 篇測試新聞。cosine 表示查詢與新聞向量的相似度。")
    tabs = st.tabs(["並排 Top 10", "新聞名次對照"])
    download_rows = []
    with tabs[0]:
        for col, method in zip(st.columns(len(results), gap="medium"), results):
            with col:
                st.markdown("#### " + SEARCH_METHOD_NAMES[method])
                setting = "比較基準 · 詞項權重" if method == "tfidf" else model_setting_description(options[selected_runs[method]][method])
                st.caption(setting)
                ranked, status = results[method]
                if not status["valid_query"]:
                    st.info("查詢沒有可用向量，無法產生排名。")
                    continue
                top10 = ranked[:10]
                if matched:
                    hits = sum(row["id"] in relevant for row in top10)
                    st.caption(f"前十名有 {hits} 篇符合「{TOPICS[matched['topic']]}」主題。")
                frame = pd.DataFrame([
                    {
                        "名次": row["rank"],
                        "新聞 ID": row["id"],
                        "cosine": row["score"],
                        "主題": "、".join(engine.by_id[row["id"]]["topics"]) or "無標籤",
                        "標題": row["title"],
                    }
                    for row in top10
                ])
                st.dataframe(frame, hide_index=True, width="stretch",
                             column_config={"名次": st.column_config.NumberColumn(width=42),
                                            "新聞 ID": st.column_config.TextColumn(width=61),
                                            "cosine": st.column_config.NumberColumn(format="%.4f", width=58),
                                            "主題": st.column_config.TextColumn(width=86, help="每篇新聞的 Reuters 主題標籤。"),
                                            "標題": st.column_config.TextColumn(width=200)})
                download_rows.extend({"query": query, "method": method, "model": method_labels[method],
                                      "run": selected_runs[method], **row} for row in top10)
    union = list(dict.fromkeys(row["id"] for ranked, _ in results.values() for row in ranked[:10]))
    with tabs[1]:
        ranks = {method: {r["id"]: r["rank"] for r in ranked} for method, (ranked, _) in results.items()}
        rows = [{"新聞 ID": key, "標題": engine.by_id[key]["title"],
                 **({"符合主題": key in relevant} if matched else {}),
                 **{method_labels[method]: rank.get(key) for method, rank in ranks.items()}} for key in union]
        st.caption("取各方法 Top 10 的聯集，顯示各自完整排名中的實際名次。空值表示該方法未能為這篇新聞排名。")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    baseline, _ = formal_evaluation(signature)
    evaluations = {"tfidf": baseline, **{method: saved_runs[run] for method, run in selected_runs.items()
                                        if method != "tfidf"}}
    models = {method: (engines[run], signatures[run]) for method, run in selected_runs.items()}
    search_effects(evaluations, bank, matched, models, results, relevant, report)
    embedding_weighting_panel(report, selected_runs, options, models, evaluations, matched)
    with st.expander("查詢詞與詞向量涵蓋"):
        rows = []
        for method, (_, status) in results.items():
            missing = status["oov"]
            if method != "tfidf":
                selected_engine = engines[selected_runs[method]]
                positions = selected_engine.space.positions
                vectors = selected_engine.word_matrices[method]
                missing = [token for token in sorted(set(status["tokens"]))
                           if token not in positions or not vectors[positions[token]].any()]
            rows.append({"模型": method_labels[method], "有效查詢": status["valid_query"],
                         "TF-IDF 詞表命中": f'{status["known_tokens"]}/{status["total_tokens"]}',
                         "詞表外／無向量": "、".join(missing) or "無",
                         "詞向量權重涵蓋率": status["embedding_weight_coverage"] if method != "tfidf" else None,
                         "有效候選新聞": status["valid_documents"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("權重涵蓋率是可用詞向量的 TF-IDF 權重比例。零向量不參與排名，其餘候選保留零分與負分。")
    if download_rows:
        st.download_button("下載 Top 10", csv_bytes(pd.DataFrame(download_rows)), "search-top10.csv", "text/csv")
    search_final_conclusion(bank, evaluations, models, selected_runs, method_labels, report)
    if union:
        st.divider()
        st.subheader("閱讀結果新聞")
        key = st.selectbox("從 Top 10 聯集中選擇", union,
                           format_func=lambda key: f'{key} · {engine.by_id[key]["title"][:110]}', key="result_document")
        shared = sorted(set(terms(query)) & set(terms(document_text(engine.by_id[key]))))
        st.caption("共同查詢詞：" + ("、".join(shared) if shared else "無。這篇可能由相近的詞向量召回。"))
        read_news(engine.by_id[key])


@st.cache_data(show_spinner=False)
def saved_evaluation(signature, query_signature, source_signature):
    output = ROOT / "reports" / "evaluation-v1"
    if (output / "manifest.json").exists():
        result, manifest = checked_evaluation(signature)
        if manifest["queries_sha256"] == query_signature:
            return result, True
    rows, summary, _ = evaluate(resources(signature)[0])
    return {"per_query": rows, "summary": summary}, False


def formal_evaluation(signature):
    sources = tuple((p.name, sha256(p)) for p in sorted((ROOT / "ir_hw4").glob("*.py")))
    return saved_evaluation(signature, sha256(ROOT / "resources" / "queries.json"), sources)


def evaluation_page(engine, signature):
    heading("04 / EVALUATION", "檢索效果評估", "以固定十個查詢與 Reuters 主題標籤，檢驗文件排序是否改善。")
    view = st.radio("評估內容", ("正式結果", "參數實驗"), horizontal=True, key="evaluation_mode")
    if view == "參數實驗":
        experiment_panel(signature)
        return
    st.info("相關性來源：Reuters topics。這是資料集提供的主題代理，並非針對這十個查詢逐篇人工判讀。")
    result, saved = formal_evaluation(signature)
    selection = st.selectbox("檢視結果", ["全部查詢平均", *[q["qid"] for q in query_bank()]], key="evaluation_query",
                             format_func=lambda key: key if key == "全部查詢平均" else next(f'{key} · {q["text"]}' for q in query_bank() if q["qid"] == key))
    average = selection == "全部查詢平均"
    rows = result["summary"] if average else [r for r in result["per_query"] if r["qid"] == selection]
    st.caption("每個查詢先各自評分，再以相同的有效查詢集合取平均。P@10 的分母固定為 10；MRR 使用完整返回排名。")
    for col, (metric, label) in zip(st.columns(3, gap="medium"), METRICS.items()):
        with col:
            display_label = "RR" if metric == "mrr" and not average else label
            frame = pd.DataFrame([{"方法": METHODS[r["method"]], "分數": r[metric]} for r in rows if r[metric] is not None])
            st.markdown("#### " + display_label)
            if not frame.empty:
                bars = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("方法:N", sort=list(METHODS.values()), title=None, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y("分數:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                    color=alt.Color("方法:N", scale=alt.Scale(domain=list(METHODS.values()), range=COLORS), legend=None),
                    tooltip=["方法", alt.Tooltip("分數:Q", format=".4f")]).properties(height=260)
                labels = alt.Chart(frame).mark_text(dy=-8).encode(x=alt.X("方法:N", sort=list(METHODS.values())), y="分數:Q", text=alt.Text("分數:Q", format=".4f"))
                st.altair_chart(bars + labels, width="stretch")
            else:
                st.info("此查詢沒有參考正例，未納入平均。")
    compact = [{"方法": METHODS[r["method"]], **{("RR" if key == "mrr" and not average else name): r[key] for key, name in METRICS.items()},
                **({"查詢數": f'{r["queries_included"]}/{r["queries_total"]}'} if average else {"已知相關新聞": r["relevant_total"]})} for r in rows]
    st.dataframe(pd.DataFrame(compact), hide_index=True, width="stretch")
    if average:
        baseline = next(r for r in rows if r["method"] == "tfidf")
        st.markdown("**相對 TF-IDF 的差值**")
        for row in rows:
            if row["method"] != "tfidf" and row["p10"] is not None:
                st.write(f'{METHODS[row["method"]]}：P@10 {row["p10"]-baseline["p10"]:+.4f}、MRR {row["mrr"]-baseline["mrr"]:+.4f}、nDCG@10 {row["ndcg10"]-baseline["ndcg10"]:+.4f}')
        st.caption("差值只描述本次固定查詢與設定，不代表統計顯著或模型普遍優劣。")
    else:
        chosen = next(q for q in query_bank() if q["qid"] == selection)
        st.write(f'主題：{chosen["topic"]} · {chosen["intent"]}。具有這個 topics 標籤的測試新聞視為相關。')
        rankings = search_results(chosen["text"], signature)
        comparison = [{"方法": METHODS[method], **r,
                       "主題相關": chosen["topic"] in engine.by_id[r["id"]]["topics"]}
                      for method in METHODS for r in rankings[method][0][:10]]
        st.dataframe(pd.DataFrame(comparison).rename(columns={"rank": "名次", "id": "新聞 ID", "title": "標題", "score": "cosine"}), hide_index=True, width="stretch")
    with st.expander("十個查詢與評估限制", expanded=False):
        st.dataframe(pd.DataFrame(query_bank()).rename(columns={"qid": "查詢", "topic": "目標主題", "text": "文字", "intent": "意圖"}), hide_index=True, width="stretch")
        st.write("主題相關不代表新聞滿足查詢的每個細節。資料類別不平衡，兩個同主題查詢也不是獨立樣本。此處只有單一 seed 與一組訓練參數，不預設 embedding 或 Skip-Gram 一定較好。")
        st.write("查詢無向量時返回空排名、指標為 0；沒有參考正例才排除平均，並揭露分母。")
    a, b = st.columns(2)
    a.download_button("下載逐查詢指標", csv_bytes(pd.DataFrame(result["per_query"])), "metrics_per_query.csv", "text/csv")
    b.download_button("下載平均指標", csv_bytes(pd.DataFrame(result["summary"])), "metrics_summary.csv", "text/csv")
    st.caption("畫面讀取已驗證的實驗結果。" if saved else "目前結果在記憶體重新計算，沒有覆寫保存的實驗。")
    document_case_panel(engine, selection if not average else None)
    st.divider()
    st.subheader("完整報告")
    st.write("整合資料與設定、十詞 Top-5 與解讀、全部查詢指標、改善與失敗案例、原文證據及限制。")
    report_downloads("evaluation")


def render():
    st.markdown("""<style>
    .block-container {max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -.035em; font-size: 2.35rem !important;}
    h2 {letter-spacing: -.02em;}
    [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DDE3DB; padding: 1rem 1.2rem; border-radius: 12px;}
    [data-testid="stSidebar"] {border-right: 1px solid #DDE3DB;}
    [data-testid="stCaptionContainer"] {color: #617365;}
    .st-key-paired_query_table td:first-child {white-space: nowrap; font-weight: 600;}
    .weighting-selection {display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .6rem; margin: 0 0 .8rem;}
    .weighting-delta {margin-left: auto; color: #52695D; font-size: .82rem; font-variant-numeric: tabular-nums;}
    .weighting-pair {display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 8.5rem), 1fr)); gap: .65rem;}
    .weighting-score {padding: .8rem; border: 1px solid #DEE5DC; border-radius: 10px; background: #EEF3EE; min-width: 0;}
    .weighting-weighted {background: #F5F0E8; border-color: #E9E0D4;}
    .weighting-mode {font-size: .86rem; font-weight: 650;}
    .weighting-scoreline {display: grid; gap: .05rem; margin: .3rem 0 .4rem;}
    .weighting-scoreline strong {font-size: 1.7rem; line-height: 1.2; font-variant-numeric: tabular-nums;}
    .weighting-scoreline span {font-size: .78rem; color: #52695D;}
    .weighting-rank {min-height: 2.3rem; margin-bottom: .35rem; font-size: .8rem;}
    .weighting-rank small {display: block; color: #52695D; font-size: .74rem;}
    .weighting-metrics {display: flex; flex-wrap: wrap; gap: .15rem .65rem; margin-bottom: .5rem; color: #52695D; font-size: .74rem; font-variant-numeric: tabular-nums;}
    .weighting-extreme {display: grid; grid-template-columns: 1fr auto; align-items: center; gap: .3rem; padding: .5rem 0 .2rem; border-top: 1px solid #D8DFD5;}
    .weighting-extreme strong {font-size: .87rem; font-variant-numeric: tabular-nums;}
    .weighting-extreme code {grid-column: 1 / -1; width: fit-content; overflow-wrap: anywhere;}
    .weighting-tie {grid-column: 1 / -1; color: #52695D; font-size: .75rem;}
    .weighting-badge {width: fit-content; border-radius: 6px; padding: .14rem .44rem; font-size: .75rem; font-weight: 650; white-space: nowrap;}
    .weighting-current {background: #E6EDF7; color: #244F82;}
    .weighting-high {background: #DDECE1; color: #245841;}
    .weighting-low {background: #F5E3D7; color: #874126;}
    .stButton button, .stDownloadButton button {border-radius: 8px;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("## Reuters Lab")
    st.sidebar.caption("IR HW4 · 新聞檢索實驗")
    st.sidebar.divider()
    page = st.sidebar.radio("工作頁", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("固定比較")
    st.sidebar.markdown("TF-IDF  \n加權 CBOW  \n加權 Skip-Gram")
    st.sidebar.caption("ModApte · train → test")
    if not (RUN / "manifest.json").exists():
        st.info("資料與模型尚未準備完成，請先執行 scripts/setup_hw4.ps1。")
        return
    try:
        signature = sha256(RUN / "manifest.json")
        engine, manifest, documents, data_manifest = resources(signature)
        with st.sidebar:
            st.divider()
            report_downloads("sidebar", compact=True)
        if page == PAGES[0]:
            overview(engine, manifest, documents, data_manifest)
        elif page == PAGES[1]:
            word_analysis(signature)
        elif page == PAGES[2]:
            news_search(engine, signature)
        else:
            evaluation_page(engine, signature)
    except (ValueError, OSError) as error:
        st.error("資料或模型版本檢查未通過，請查看專案驗證紀錄。")
        st.exception(error)
