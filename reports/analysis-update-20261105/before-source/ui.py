"""Four read-only pages. UI never downloads, trains or rewrites experiments."""
from collections import Counter
import io
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from .config import ROOT, RUN, METHODS, TOPICS, ANALYSIS_WORDS, README_URL
from .dataset import load_data
from .evaluation import query_bank, evaluate
from .io import read_json, sha256
from .retrieval import neighbors, unit_rows
from .text import terms, document_text, normalize
from .training import load_engine

PAGES = ["資料概況", "詞向量分析", "新聞搜尋與比較", "效果評估"]
COLORS = ["#173F50", "#C66B3D", "#749385"]
METRICS = {"p10": "Precision@10", "mrr": "MRR", "ndcg10": "nDCG@10"}


@st.cache_resource(show_spinner="正在載入 Reuters 與兩組詞向量…")
def resources(signature):
    engine, manifest = load_engine()
    documents, data_manifest = load_data()
    return engine, manifest, documents, data_manifest


@st.cache_data(show_spinner=False)
def search_results(query, signature):
    engine = resources(signature)[0]
    return {method: engine.rank(query, method) for method in METHODS}


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


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


def word_analysis(engine, manifest, signature):
    heading("02 / WORDS", "詞向量分析", "找出出現在相似上下文的詞，再回到新聞內容理解它們的關係。")
    def fill_word():
        st.session_state["word_input"] = st.session_state["word_preset"]
    st.selectbox("範例詞", ANALYSIS_WORDS, index=5, key="word_preset", on_change=fill_word)
    if "word_input" not in st.session_state:
        st.session_state["word_input"] = "oil"
    with st.form("word_form"):
        st.text_input("分析一個英文詞", key="word_input")
        submitted = st.form_submit_button("查看 Top-5 近鄰", type="primary")
    if submitted:
        tokens = terms(st.session_state["word_input"])
        st.session_state["word_submitted"] = tokens[0] if len(tokens) == 1 else ""
    word = st.session_state.get("word_submitted", "oil")
    if not word:
        st.info("請輸入一個前處理後有效的英文詞。")
        return
    if terms(st.session_state["word_input"]) != [word]:
        st.caption(f"目前仍顯示已提交的 {word}；修改後請按「查看 Top-5 近鄰」。")
    st.subheader(f"{word} 的 Top-5 近鄰")
    all_rows = []
    for col, name in zip(st.columns(2, gap="large"), ("cbow", "skipgram")):
        with col:
            st.markdown("#### " + ("CBOW" if name == "cbow" else "Skip-Gram"))
            result = neighbors(*engine.models[name], word)
            if not result:
                st.info("這個詞沒有可用詞向量，可能未出現在訓練集或未達 min_count=2。")
                continue
            frame = pd.DataFrame(result).rename(columns={"word": "鄰近詞", "cosine": "cosine"})
            chart = alt.Chart(frame).mark_bar(color=COLORS[0 if name == "cbow" else 1], cornerRadiusEnd=3).encode(
                y=alt.Y("鄰近詞:N", sort=frame["鄰近詞"].tolist(), title=None),
                x=alt.X("cosine:Q", scale=alt.Scale(domain=[-1, 1]), title="原始 100 維 cosine"),
                tooltip=["鄰近詞", alt.Tooltip("cosine:Q", format=".6f")]).properties(height=205)
            st.altair_chart(chart, width="stretch")
            st.dataframe(frame, hide_index=True, width="stretch", column_config={"cosine": st.column_config.NumberColumn(format="%.4f")})
            all_rows.extend({"分析詞": word, "模型": name, "名次": i, **r} for i, r in enumerate(result, 1))
    st.info("鄰近詞可能是同義、反義、共現詞，或出現在相似新聞情境。向量接近不保證能改善文件檢索；請到「效果評估」比較實際排名。")
    if all_rows:
        st.download_button("下載這個詞的近鄰", csv_bytes(pd.DataFrame(all_rows)), f"neighbors-{word}.csv", "text/csv")
    with st.expander("閱讀訓練新聞中的使用情境"):
        for row in training_context(word, signature):
            st.markdown(f'**{row["id"]} · {row["title"]}**')
            st.write(row["excerpt"])
        st.caption("片段是使用情境範例，不是人工判定近鄰合理性的標準答案。")
    with st.expander("比較兩個詞與 PCA 輔助圖"):
        a, b = st.columns(2)
        first = a.text_input("詞 A", value="profit", key="pair_a")
        second = b.text_input("詞 B", value="loss", key="pair_b")
        pair = [terms(first), terms(second)]
        if all(len(t) == 1 for t in pair):
            for name, (words, vectors) in engine.models.items():
                positions = {term: i for i, term in enumerate(words)}
                if all(t[0] in positions for t in pair):
                    units = unit_rows(np.array([vectors[positions[t[0]]] for t in pair], dtype=np.float64))
                    st.write(f'{"CBOW" if name == "cbow" else "Skip-Gram"}：{pair[0][0]} ／ {pair[1][0]} cosine = {float(units[0] @ units[1]):.6f}')
                else:
                    st.caption(f"{name}：至少一詞沒有詞向量。")
        st.caption("例如 profit／loss 即使接近，也可能反映相似的財務句式，不能視為同義詞。")
        if st.checkbox("顯示 PCA 圖", key="show_pca"):
            name = st.selectbox("投影模型", ["cbow", "skipgram"], format_func=lambda x: "CBOW" if x == "cbow" else "Skip-Gram")
            words, vectors = engine.models[name]
            positions = {term: i for i, term in enumerate(words)}
            selected = list(dict.fromkeys([*ANALYSIS_WORDS, word, *[r["word"] for r in neighbors(words, vectors, word)]]))
            selected = [term for term in selected if term in positions]
            if len(selected) >= 3:
                matrix = unit_rows(np.array([vectors[positions[term]] for term in selected], dtype=np.float64))
                matrix -= matrix.mean(axis=0)
                u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
                xy = u[:, :2] * singular[:2]
                variance = float(sum(singular[:2]**2) / sum(singular**2)) if sum(singular**2) else 0.
                frame = pd.DataFrame({"詞": selected, "PC1": xy[:, 0], "PC2": xy[:, 1]})
                points = alt.Chart(frame).mark_circle(size=70, color=COLORS[1]).encode(x="PC1:Q", y="PC2:Q", tooltip=["詞", "PC1", "PC2"])
                labels = alt.Chart(frame).mark_text(align="left", dx=7, dy=-6).encode(x="PC1:Q", y="PC2:Q", text="詞:N")
                st.altair_chart((points + labels).properties(height=360), width="stretch")
                st.caption(f"PCA 以這組詞的 L2 詞向量顯示主要差異，前兩軸保留 {variance:.1%} 的變異。座標不是正確率；投影有資訊損失，近鄰與排名仍使用原始 100 維向量。")
    with st.expander("模型訓練資訊"):
        st.json(manifest["models"], expanded=False)


def news_search(engine, signature):
    heading("03 / RETRIEVAL", "新聞搜尋與比較", "同一個查詢，觀察三種文件表示如何改變前十名。")
    bank = query_bank()
    presets = {row["qid"]: row for row in bank}
    def fill_query():
        st.session_state["search_input"] = presets[st.session_state["query_preset"]]["text"]
    st.selectbox("範例查詢", list(presets), index=4, key="query_preset", on_change=fill_query,
                 format_func=lambda key: f'{key} · {presets[key]["text"]}（{presets[key]["intent"]}）')
    if "search_input" not in st.session_state:
        st.session_state["search_input"] = bank[4]["text"]
    with st.form("search_form"):
        st.text_input("英文查詢", key="search_input", placeholder="例如 crude oil production")
        submitted = st.form_submit_button("搜尋並比較", type="primary")
    if submitted:
        st.session_state["submitted_query"] = st.session_state["search_input"]
    query = st.session_state.get("submitted_query", bank[4]["text"])
    if query != st.session_state["search_input"]:
        st.caption(f"目前結果屬於「{query}」。修改後請按「搜尋並比較」。")
    results = search_results(query, signature)
    st.subheader(f"搜尋結果：{query or '（空白查詢）'}")
    st.caption("候選為完整 3,299 篇測試新聞；cosine 不是正確率。零向量不參與排名，其餘候選保留零分與負分。")
    tabs = st.tabs(["並排 Top 10", "新聞名次對照"])
    download_rows = []
    with tabs[0]:
        for col, method in zip(st.columns(3, gap="medium"), METHODS):
            with col:
                st.markdown("#### " + METHODS[method])
                ranked, status = results[method]
                if not status["valid_query"]:
                    st.info("查詢沒有可用向量，無法產生排名。")
                    continue
                frame = pd.DataFrame(ranked[:10]).rename(columns={"rank": "名次", "id": "新聞 ID", "score": "cosine", "title": "標題"})
                st.dataframe(frame, hide_index=True, width="stretch", height=425,
                             column_config={"cosine": st.column_config.NumberColumn(format="%.4f"),
                                            "標題": st.column_config.TextColumn(width="large")})
                download_rows.extend({"query": query, "method": method, **row} for row in ranked[:10])
    union = list(dict.fromkeys(row["id"] for method in METHODS for row in results[method][0][:10]))
    with tabs[1]:
        ranks = {method: {r["id"]: r["rank"] for r in results[method][0]} for method in METHODS}
        rows = [{"新聞 ID": key, "標題": engine.by_id[key]["title"],
                 **{METHODS[method]: ranks[method].get(key) for method in METHODS}} for key in union]
        st.caption("取三種方法 Top 10 的聯集，顯示各自完整排名中的實際名次。空值代表該方法沒有可用向量。")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("查詢詞與詞向量涵蓋"):
        rows = []
        for method in METHODS:
            status = results[method][1]
            rows.append({"方法": METHODS[method], "有效查詢": status["valid_query"],
                         "訓練詞表命中": f'{status["known_tokens"]}/{status["total_tokens"]}',
                         "詞向量權重涵蓋率": status["embedding_weight_coverage"] if method != "tfidf" else None,
                         "有效候選新聞": status["valid_documents"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("OOV：" + ("、".join(results["tfidf"][1]["oov"]) or "無"))
        st.caption("權重涵蓋率是可用詞向量的 TF-IDF 權重比例，不是檢索正確率。")
    if download_rows:
        st.download_button("下載三方法 Top 10", csv_bytes(pd.DataFrame(download_rows)), "search-top10.csv", "text/csv")
    if union:
        st.divider()
        st.subheader("閱讀結果新聞")
        key = st.selectbox("從三方法 Top 10 聯集中選擇", union,
                           format_func=lambda key: f'{key} · {engine.by_id[key]["title"][:110]}', key="result_document")
        shared = sorted(set(terms(query)) & set(terms(document_text(engine.by_id[key]))))
        st.caption("共同查詢詞：" + ("、".join(shared) if shared else "無。這篇可能由相近的詞向量召回。"))
        read_news(engine.by_id[key])


@st.cache_data(show_spinner=False)
def saved_evaluation(signature, query_signature, source_signature):
    output = ROOT / "reports" / "evaluation-v1"
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        current = manifest.get("model_manifest_sha256") == signature and manifest.get("queries_sha256") == query_signature
        current = current and all(sha256(ROOT / path) == digest for path, digest in manifest.get("source_sha256", {}).items())
        if current:
            for path, digest in manifest["artifacts_sha256"].items():
                if sha256(output / path) != digest:
                    raise ValueError("Saved evaluation artifact mismatch")
            return read_json(output / "results.json"), True
    rows, summary, _ = evaluate(resources(signature)[0])
    return {"per_query": rows, "summary": summary}, False


def evaluation_page(engine, signature):
    heading("04 / EVALUATION", "檢索效果評估", "以固定十個查詢與 Reuters 主題標籤，檢驗文件排序是否改善。")
    st.info("相關性來源：Reuters topics。這是資料集提供的主題代理，並非針對這十個查詢逐篇人工判讀。")
    source_signature = tuple((p.name, sha256(p)) for p in sorted((ROOT / "ir_hw4").glob("*.py")))
    result, saved = saved_evaluation(signature, sha256(ROOT / "resources" / "queries.json"), source_signature)
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
    report = ROOT / "reports" / "evaluation-v1" / "REPORT.md"
    if saved and report.exists():
        st.download_button("下載完整實驗報告", report.read_bytes(), "HW4_REPORT.md", "text/markdown")
    st.caption("畫面讀取已驗證的實驗結果。" if saved else "目前結果在記憶體重新計算，沒有覆寫保存的實驗。")


def render():
    st.markdown("""<style>
    .block-container {max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -.035em; font-size: 2.35rem !important;}
    h2 {letter-spacing: -.02em;}
    [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DDE3DB; padding: 1rem 1.2rem; border-radius: 12px;}
    [data-testid="stSidebar"] {border-right: 1px solid #DDE3DB;}
    [data-testid="stCaptionContainer"] {color: #617365;}
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
        if page == PAGES[0]:
            overview(engine, manifest, documents, data_manifest)
        elif page == PAGES[1]:
            word_analysis(engine, manifest, signature)
        elif page == PAGES[2]:
            news_search(engine, signature)
        else:
            evaluation_page(engine, signature)
    except (ValueError, OSError) as error:
        st.error("資料或模型版本檢查未通過，請查看專案驗證紀錄。")
        st.exception(error)
