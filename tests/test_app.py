import csv
import json
from pathlib import Path
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    return at


def navigate(app, page):
    app.sidebar.radio(key="page").set_value(page).run(timeout=90)
    assert not app.exception
    return app


def submit_search(app):
    next(button for button in app.button if button.label == "搜尋並比較").click().run(timeout=90)


@pytest.fixture
def word_app(app):
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN
    from ir_hw4.experiments import completed_experiments

    if DEFAULT_MODEL_RUN not in completed_experiments():
        pytest.skip("The 100-dimension, window-5 run is required for these word examples")
    navigate(app, "詞向量分析")
    app.selectbox(key="word_cbow_run").select(DEFAULT_MODEL_RUN)
    app.selectbox(key="word_skipgram_run").select(DEFAULT_MODEL_RUN).run()
    return app


def test_overview_has_correct_modapte_counts_and_four_pages(app):
    assert app.sidebar.radio(key="page").options == ["資料概況", "詞向量分析", "新聞搜尋與比較", "效果評估"]
    assert app.metric[0].value == "9,603" and app.metric[1].value == "3,299"
    app.text_input(key="term_filter").set_value("zzzzunknown").run()
    assert not app.exception and app.dataframe[-1].value.empty
    app.text_input(key="term_filter").set_value("oil").run()
    assert not app.dataframe[-1].value.empty


def test_word_neighbors_oov_and_pca(word_app):
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN

    app = word_app
    assert len(app.dataframe[0].value) == 5 and len(app.dataframe[1].value) == 5
    assert app.radio(key="word_map_view").value == "Top-5 相似度"
    app.selectbox(key="word_preset").select("loss").run()
    app.button[0].click().run()
    app.radio(key="word_map_view").set_value("PCA 空間").run()
    assert not app.exception
    charts = app.get("vega_lite_chart")
    assert len(charts) == 4
    for chart in charts[-2:]:
        for layer in json.loads(chart.proto.spec)["layer"]:
            assert layer["encoding"]["x"]["axis"]["title"] == "X（PC1）"
            assert layer["encoding"]["y"]["axis"]["title"] == "Y（PC2）"
    suffix = f"loss_{DEFAULT_MODEL_RUN}"
    assert app.selectbox(key=f"pca_first_cbow_{suffix}").value == "loss"
    assert app.selectbox(key=f"pca_second_cbow_{suffix}").value == "profit"
    assert "net" not in app.selectbox(key=f"pca_first_cbow_{suffix}").options
    assert "profit" not in app.selectbox(key=f"pca_first_skipgram_{suffix}").options
    distances = [element.value for element in app.markdown if "100 維距離" in element.value]
    assert any("0.7190" in text and "0.7497" in text and "0.6260" in text for text in distances)
    assert any("0.7679" in text and "0.6814" in text and "0.3281" in text for text in distances)
    app.selectbox(key=f"pca_first_skipgram_{suffix}").select("oper").run()
    app.selectbox(key=f"pca_second_skipgram_{suffix}").select("net").run()
    assert not app.exception
    distances = [element.value for element in app.markdown if "100 維距離" in element.value]
    assert any("0.7190" in text and "0.7497" in text for text in distances)
    assert any("0.6717" in text and "0.8103" in text and "0.8031" in text for text in distances)
    app.radio(key="word_map_view").set_value("Top-5 相似度").run()
    assert not app.exception and any("長條從 0 起算" in element.value for element in app.caption)
    charts = app.get("vega_lite_chart")
    assert len(charts) == 4
    for chart, table in zip(charts[-2:], app.dataframe[:2]):
        spec = json.loads(chart.proto.spec)
        bars = spec["layer"][0]
        y = bars["encoding"]["y"]
        assert spec["autosize"]["type"] == "fit-x" and spec["height"]["step"] > bars["mark"]["size"]
        assert y["axis"]["labelOverlap"] is False
        assert y["sort"] == [f"{rank} · {word}" for rank, word in zip(table.value["名次"], table.value["鄰近詞"])]
    app.text_input(key="word_input").set_value("zzzzunknown")
    app.button[0].click().run()
    assert not app.exception
    assert any("沒有可用詞向量" in element.value for element in app.info)


def test_tsne_shows_original_neighbors_and_evidence_on_both_maps(word_app):
    app = word_app
    app.selectbox(key="word_preset").select("oil").run()
    app.button[0].click().run()
    original = [element.value for element in app.dataframe[:2]]
    app.radio(key="word_map_view").set_value("t-SNE 空間").run()
    assert not app.exception
    assert all(element.value.equals(table) for element, table in zip(app.dataframe[:2], original))
    charts = app.get("vega_lite_chart")
    assert len(charts) == 4
    for chart, table in zip(charts[-2:], original):
        spec = json.loads(chart.proto.spec)
        for layer in spec["layer"]:
            for channel, title in (("x", "X（t-SNE 1）"), ("y", "Y（t-SNE 2）")):
                axis = layer["encoding"][channel]["axis"]
                assert axis["title"] == title
                assert axis["labels"] and axis["ticks"] and axis["grid"]
        symbols = next(layer for layer in spec["layer"] if "tooltip" in layer.get("encoding", {}))
        name = symbols["data"]["name"]
        dataset = next(dataset for dataset in chart.proto.datasets if dataset.name == name)
        points = pa.ipc.open_stream(dataset.data.data).read_pandas().set_index("word")
        assert set(points.index) == {"oil", *table["鄰近詞"]} and not points[["x", "y"]].isna().any().any()
        assert points.loc[table["鄰近詞"], "cosine"].tolist() == pytest.approx(table["cosine"].tolist())
        if "kernel" in points.index:
            assert points.loc["kernel", "label"] == "4 · kernel"
            assert points.loc["kernel", "cooccurrence"] == 5
    descriptions = [element.value for element in app.markdown]
    first = descriptions.index("**1. 哪些鄰近詞具有合理的語意或使用情境關係？**")
    second = descriptions.index("**2. 哪些結果可能來自經常出現的相似新聞情境？**")
    third = descriptions.index("**3. 哪些結果不符合預期？可能原因為何？**")
    assert any("CBOW" in text and "`crude` 第 5 名" in text for text in descriptions[first:second])
    assert any("natural gas" in text and "**和 `oil` 的關係**" in text for text in descriptions[second:third])
    assert any("`natural`" in item.value and "同篇共現 83 篇" in item.value for item in app.caption)
    assert any("CBOW" in text and "`kernel` 第 4 名" in text for text in descriptions[third:])
    assert any("palm kernel" in text and "為什麼排進前五名會讓人意外" in text for text in descriptions[third:])
    assert any("`kernel`" in item.value and "同篇共現 5 篇" in item.value for item in app.caption)
    assert any("訓練新聞 885" in text for text in descriptions)
    pair_metrics = [text for text in descriptions if "100 維距離" in text]
    assert len(pair_metrics) == 2
    assert all("原始 cosine" in text and "2 維距離" not in text for text in pair_metrics)


def test_full_pca_projects_all_model_words_and_keeps_top_five_analysis(word_app):
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN
    from ir_hw4.experiments import experiment_paths

    app = word_app
    center = app.text_input(key="word_input").value
    original = [element.value for element in app.dataframe[:2]]
    app.radio(key="word_map_view").set_value("PCA 空間").run()
    scope = app.radio(key="word_projection_limit")
    assert scope.value == "前 5 詞" and scope.options == ["前 5 詞", "全部詞"]
    scope.set_value("全部詞").run()
    assert not app.exception
    assert all(element.value.equals(table) for element, table in zip(app.dataframe[:2], original, strict=True))
    run, _ = experiment_paths(DEFAULT_MODEL_RUN)
    charts = app.get("vega_lite_chart")[-2:]
    for method, chart, table in zip(("cbow", "skipgram"), charts, original, strict=True):
        vocabulary = json.loads((run / method / "vocabulary.json").read_text(encoding="utf-8"))
        spec = json.loads(chart.proto.spec)
        symbols = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "point")
        dataset = next(item for item in chart.proto.datasets if item.name == symbols["data"]["name"])
        points = pa.ipc.open_stream(dataset.data.data).read_pandas().sort_values("rank")
        assert len(points) == len(vocabulary) and set(points["word"]) == set(vocabulary)
        assert points.iloc[0]["word"] == center and not points[["x", "y"]].isna().any().any()
        assert points.iloc[1:6]["word"].tolist() == table["鄰近詞"].tolist()
        assert points.iloc[1:6]["cosine"].tolist() == pytest.approx(table["cosine"].tolist())

    key = f"pca_first_cbow_{center}_{DEFAULT_MODEL_RUN}"
    last_neighbor = app.selectbox(key=key).options[-1]
    app.selectbox(key=key).select(last_neighbor).run()
    assert not app.exception and app.selectbox(key=key).value == last_neighbor
    app.radio(key="word_projection_limit").set_value("前 5 詞").run()
    assert not app.exception and app.selectbox(key=key).value == center


def test_search_blank_oov_and_semantic_query(app):
    navigate(app, "新聞搜尋與比較")
    assert len(app.dataframe[0].value) == 10
    for query in ("", "zzzzunknown", "petroleum output prices"):
        app.text_input(key="search_input").set_value(query)
        submit_search(app)
        assert not app.exception
        if query != "petroleum output prices":
            assert any("沒有可用向量" in e.value for e in app.info)
        else:
            assert len(app.dataframe[0].value) == 10


def test_preset_fills_input_but_submitted_results_remain_until_search(app):
    navigate(app, "新聞搜尋與比較")
    assert app.selectbox(key="query_preset").value == "Q01"
    assert app.text_input(key="search_input").value == "corporate profit earnings"
    assert app.subheader[0].value == "搜尋結果：corporate profit earnings"
    app.radio(key="search_effect_scope").set_value("同題三種問法").run()
    paired = app.table[0].value.copy()
    app.selectbox(key="query_preset").select("Q10").run()
    assert app.text_input(key="search_input").value == "import export trade"
    assert any("目前結果屬於" in e.value for e in app.caption)
    assert app.table[0].value.equals(paired)
    submit_search(app)
    assert not app.exception
    assert app.subheader[0].value == "搜尋結果：import export trade"
    assert app.table[0].value.iloc[0]["英文查詢"] == "import export trade"


def test_search_effects_report_the_submitted_query_and_only_score_benchmarks(app):
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN
    from ir_hw4.experiments import completed_experiments

    if DEFAULT_MODEL_RUN not in completed_experiments():
        pytest.skip("The default 15-epoch experiment is required for these scores")
    navigate(app, "新聞搜尋與比較")
    scope = app.radio(key="search_effect_scope")
    assert scope.options[0] == scope.value == "目前查詢"
    app.selectbox(key="search_cbow_run").select(DEFAULT_MODEL_RUN)
    app.selectbox(key="search_skipgram_run").select(DEFAULT_MODEL_RUN).run()
    initial_scores = app.table[0].value.copy()
    app.selectbox(key="query_preset").select("Q06").run()
    assert app.table[0].value.equals(initial_scores)

    submit_search(app)
    assert not app.exception
    assert app.table[0].value.iloc[0].tolist()[1:] == ["0.4784", "0.6215（+0.1431）", "1.0000（+0.5216）"]
    assert app.table[0].value.iloc[1].tolist()[1:] == ["0.4000", "0.7000（+0.3000）", "1.0000（+0.6000）"]
    assert "符合主題" in app.dataframe[3].value.columns

    app.text_input(key="search_input").set_value("not petroleum output prices")
    submit_search(app)
    assert not app.exception and not app.table
    assert any("沒有預先指定的相關主題" in message.value for message in app.info)
    assert any("可提交一個範例查詢" in message.value for message in app.info)
    assert "符合主題" not in app.dataframe[3].value.columns
    app.radio(key="search_effect_scope").set_value("關鍵字 10 題平均").run()
    assert app.table[0].value.iloc[0].tolist()[1:] == ["0.8840", "0.8828（-0.0013）", "0.9191（+0.0351）"]
    assert any("首篇符合主題的新聞由" in element.value for element in app.markdown)


def test_embedding_weighting_comparison_shows_extrema_and_current_models(app):
    from ir_hw4.embedding_weighting import COMPARISON

    if not (COMPARISON / "manifest.json").exists():
        pytest.skip("The offline weighting comparison is required for these scores")
    navigate(app, "新聞搜尋與比較")
    assert app.radio(key="embedding_weighting_scope").value == "目前查詢"
    winners = next(element.value for element in app.table if "最高欄位" in element.value)
    assert winners.loc[winners["架構"] == "Skip-Gram"].values.tolist() == [
        ["Skip-Gram", "單用最高", "v150-w15-e15", "0.9474（第 1 名）", "0.9291（第 3 名）"],
        ["Skip-Gram", "加權最高", "v100-w15-e15", "0.9436（第 2 名）", "0.9526（第 1 名）"],
    ]
    app.radio(key="search_effect_scope").set_value("三種問法總覽").run()
    overview = app.table[0].value
    assert overview.loc[overview["指標"] == "nDCG@10"].values.tolist() == [
        ["關鍵字 10／10 題", "nDCG@10", "0.8840", "0.9203（+0.0363）", "0.9291（+0.0450）"],
        ["模糊句 10／10 題", "nDCG@10", "0.5135", "0.4808（-0.0327）", "0.7012（+0.1877）"],
        ["混用 10／10 題", "nDCG@10", "0.7497", "0.8387（+0.0890）", "0.8508（+0.1011）"],
    ]
    assert overview.iloc[-1].tolist()[1:] == ["MRR", "0.9333", "0.8500（-0.0833）", "0.9333（持平）"]
    app.radio(key="search_effect_scope").set_value("目前查詢")
    app.selectbox(key="query_preset").select("Q06").run()
    assert any("**Q01 · 關鍵字**" in item.value for item in app.markdown)
    submit_search(app)
    cbow, skipgram = [element.value for element in app.html if 'class="weighting-pair"' in element.value]
    assert "0.7760</strong>" in cbow and "P@10 <b>0.8000</b>" in cbow and "RR <b>1.0000</b>" in cbow
    assert "0.9149</strong>" in skipgram and "P@10 <b>0.9000</b>" in skipgram
    assert "並列最高" in cbow

    app.selectbox(key="query_preset").select("Q09-F").run()
    submit_search(app)
    assert any("**Q09-F · 模糊句**" in item.value for item in app.markdown)
    cbow = next(element.value for element in app.html if 'class="weighting-pair"' in element.value)
    assert "0.0734</strong>" in cbow and "RR <b>0.0588</b>" in cbow and "RR <b>0.1429</b>" in cbow
    assert "參數排名 <strong>全部同分</strong><small>共 18 組</small>" in cbow
    assert app.table[0].value.iloc[0]["加權 CBOW"].startswith("0.0734")

    app.radio(key="embedding_weighting_metric").set_value("mrr").run()
    cbow = next(element.value for element in app.html if 'class="weighting-pair"' in element.value)
    unweighted, weighted = cbow.split('class="weighting-score weighting-weighted"')
    assert "<span>RR</span><strong>0.0588</strong>" in unweighted
    assert "參數排名 <strong>第 2 名</strong>" in unweighted
    assert "<span>RR</span><strong>0.1429</strong>" in weighted
    assert "最高</span><strong>0.1429</strong><code>v300-w15-e5</code>" in weighted

    app.radio(key="embedding_weighting_scope").set_value("關鍵字 10 題平均").run()
    assert app.radio(key="embedding_weighting_metric").value == "mrr"
    cbow = next(element.value for element in app.html if 'class="weighting-pair"' in element.value)
    assert "<span>MRR</span><strong>0.9000</strong>" in cbow

    app.radio(key="embedding_weighting_metric").set_value("p10").run()
    _, skipgram = [element.value for element in app.html if 'class="weighting-pair"' in element.value]
    assert "<span>平均 Precision@10</span><strong>0.9300</strong>" in skipgram
    assert "加權差值 -0.0200" in skipgram
    grids = [element.value for element in app.dataframe if "加權 Precision@10" in element.value]
    selected = grids[1].set_index("參數").loc["v100-w15-e15"]
    assert selected["加權 Precision@10"] == "0.9400（第 1 名）" and selected["加權差值"] == "-0.0100"
    app.radio(key="embedding_weighting_metric").set_value("ndcg10").run()

    for scope, suffix, before, after in (("模糊句 10 題平均", "F", "0.4995", "0.4808"),
                                         ("混用 10 題平均", "M", "0.8562", "0.8387")):
        app.radio(key="embedding_weighting_scope").set_value(scope).run()
        questions = next(element.value for element in app.table if "題號" in element.value)
        assert questions["題號"].tolist() == [f"Q{index:02d}-{suffix}" for index in range(1, 11)]
        cbow = next(element.value for element in app.html if 'class="weighting-pair"' in element.value)
        unweighted, weighted = cbow.split('class="weighting-score weighting-weighted"')
        assert f"<span>平均 nDCG@10</span><strong>{before}</strong>" in unweighted
        assert f"<span>平均 nDCG@10</span><strong>{after}</strong>" in weighted

    app.radio(key="embedding_weighting_scope").set_value("關鍵字 10 題平均").run()
    assert not app.exception
    cbow, skipgram = [element.value for element in app.html if 'class="weighting-pair"' in element.value]
    assert "MRR <b>" in cbow and "MRR <b>" in skipgram
    expected = (
        (cbow, "v300-w15-e5", ("0.9130", "第 1 名", "0.9130", "v300-w15-e5", "0.7675", "v100-w2-e5"),
         ("0.9203", "並列第 2 名", "0.9206", "v100-w15-e5", "0.7958", "v150-w2-e5")),
        (skipgram, "v150-w15-e15", ("0.9474", "第 1 名", "0.9474", "v150-w15-e15", "0.8395", "v150-w2-e5"),
         ("0.9291", "第 3 名", "0.9526", "v100-w15-e15", "0.8148", "v300-w2-e5")),
    )
    for card, selected, unweighted, weighted in expected:
        assert f"目前選取</span><code>{selected}</code>" in card
        panels = card.split('class="weighting-score weighting-weighted"')
        for panel, stats in zip(panels, (unweighted, weighted), strict=True):
            score, rank, high, high_params, low, low_params = stats
            assert f"<span>平均 nDCG@10</span><strong>{score}</strong>" in panel
            assert f"參數排名 <strong>{rank}</strong><small>共 18 組" in panel
            assert f"最高</span><strong>{high}</strong><code>{high_params}</code>" in panel
            assert f"最低</span><strong>{low}</strong><code>{low_params}</code>" in panel
    grids = [element.value for element in app.dataframe if "單用 nDCG@10" in element.value]
    assert [len(grid) for grid in grids] == [18, 18]
    assert any("0.7799 → 0.8415" in item.value and "第 2 名 變為 第 1 名" in item.value for item in app.markdown)

    app.selectbox(key="search_cbow_run").select("grid-d150-w2-e5")
    app.selectbox(key="search_skipgram_run").select("grid-d300-w2-e5").run()
    assert not app.exception
    cbow, skipgram = [element.value for element in app.html if 'class="weighting-pair"' in element.value]
    assert "目前選取</span><code>v150-w2-e5</code>" in cbow
    assert "目前選取</span><code>v300-w2-e5</code>" in skipgram
    for card, score in ((cbow, "0.7958"), (skipgram, "0.8148")):
        weighted = card.split('class="weighting-score weighting-weighted"')[1]
        assert f"<span>平均 nDCG@10</span><strong>{score}</strong>" in weighted
        assert "參數排名 <strong>第 18 名</strong>" in weighted


def test_search_query_variants_score_the_same_topic_as_the_keyword_query(app):
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN
    from ir_hw4.experiments import completed_experiments
    from ir_hw4.text import terms
    from ir_hw4.ui import search_query_bank

    if DEFAULT_MODEL_RUN not in completed_experiments():
        pytest.skip("The default experiment is required for these scores")
    bank = search_query_bank()
    originals = {q["qid"]: q for q in bank if q["kind"] == "keyword"}
    assert len(bank) == 3 * len(originals)
    for q in bank:
        original = originals[q["base_qid"]]
        assert q["topic"] == original["topic"]
        tokens = set(terms(q["text"]))
        if q["kind"] == "fuzzy":
            topic_words = {word for item in originals.values() if item["topic"] == q["topic"]
                           for word in terms(item["text"])}
            assert tokens.isdisjoint(topic_words)
        elif q["kind"] == "mixed":
            keywords = set(terms(original["text"]))
            assert 0 < len(tokens & keywords) < len(keywords)

    navigate(app, "新聞搜尋與比較")
    app.selectbox(key="search_cbow_run").select(DEFAULT_MODEL_RUN)
    app.selectbox(key="search_skipgram_run").select(DEFAULT_MODEL_RUN)
    app.radio(key="search_effect_scope").set_value("同題三種問法")
    app.selectbox(key="query_preset").select("Q05").run()
    submit_search(app)
    paired = app.table[0].value.copy()
    assert len(paired) == 9 and set(paired["指標"]) == {"nDCG@10", "P@10", "RR"}
    ndcg = paired.loc[paired["指標"] == "nDCG@10", ["問法", "TF-IDF", "加權 CBOW", "加權 Skip-Gram"]]
    assert ndcg.values.tolist() == [
        ["關鍵字", "1.0000", "1.0000（持平）", "1.0000（持平）"],
        ["模糊句", "0.3979", "0.5482（+0.1502）", "0.8416（+0.4436）"],
        ["混用", "0.5677", "1.0000（+0.4323）", "1.0000（+0.4323）"],
    ]
    assert any("Q05-F · 模糊句" in option for option in app.selectbox(key="query_preset").options)
    app.button(key="search_variant_fuzzy").click().run()
    assert app.selectbox(key="query_preset").value == "Q05-F"
    assert app.text_input(key="search_input").value == "How much is being pumped from underground wells?"
    submit_search(app)
    assert app.table[0].value.equals(paired)
    app.radio(key="search_effect_scope").set_value("目前查詢").run()
    assert not app.exception and not app.error
    assert app.table[0].value.iloc[0].tolist()[1:] == ["0.3979", "0.5482（+0.1502）", "0.8416（+0.4436）"]
    app.radio(key="search_effect_scope").set_value("模糊句 10 題平均").run()
    assert app.table[0].value.iloc[0].tolist()[1:] == ["0.5135", "0.5427（+0.0292）", "0.6218（+0.1083）"]


def test_search_compares_two_saved_runs_for_the_same_query(app):
    from ir_hw4.embedding_weighting import COMPARISON
    from ir_hw4.evaluation import query_bank
    from ir_hw4.experiment_ui import REFERENCE_RUNS
    from ir_hw4.experiments import completed_experiments, read_experiment

    cbow_run, skipgram_run = "grid-d150-w2-e15", "grid-d150-w5-e15"
    finished = set(completed_experiments())
    if not {cbow_run, skipgram_run}.issubset(finished):
        pytest.skip("The two saved 15-epoch runs are required for this comparison")
    query = query_bank()[0]
    navigate(app, "新聞搜尋與比較")
    standalone_ready = set(REFERENCE_RUNS).issubset(finished) and (COMPARISON / "manifest.json").is_file()
    standalone_best_runs = {"cbow": "grid-d300-w15-e5", "skipgram": "grid-d150-w15-e15"}
    search_options = {}
    for method, window in (("cbow", 2), ("skipgram", 5)):
        picker = app.selectbox(key=f"search_{method}_run")
        labels = picker.options
        search_options[method] = labels
        assert any(label.startswith(f"150 維 · 窗口 {window} · 15 輪") for label in labels)
        assert all("30 輪" not in label for label in labels)
        if standalone_ready:
            assert picker.value == standalone_best_runs[method]
            assert any(label.endswith((" (最高)", " (並列最高)")) for label in labels)
            assert any(label.endswith((" (最低)", " (並列最低)")) for label in labels)
    lexical, _, original_skipgram = [element.value for element in app.dataframe[:3]]
    app.selectbox(key="query_preset").select("Q10").run()
    app.selectbox(key="search_cbow_run").select(cbow_run).run()
    assert not app.exception and not app.error
    assert app.dataframe[0].value.equals(lexical)
    assert app.dataframe[2].value.equals(original_skipgram)
    cbow = app.dataframe[1].value
    app.selectbox(key="search_skipgram_run").select(skipgram_run).run()
    assert not app.exception and not app.error
    assert app.subheader[0].value == f"搜尋結果：{query['text']}"
    assert app.dataframe[0].value.equals(lexical) and app.dataframe[1].value.equals(cbow)
    labels = {column.split("（", 1)[0] for column in app.dataframe[3].value.columns}
    assert {"CBOW-v150-w2-e15", "Skip-Gram-v150-w5-e15"} <= labels
    app.radio(key="search_effect_scope").set_value("關鍵字 10 題平均").run()
    effect = app.table[0].value.iloc[0]
    assert effect["TF-IDF"] == "0.8840"
    for index, method, name, label in ((1, "cbow", cbow_run, "加權 CBOW"),
                                       (2, "skipgram", skipgram_run, "加權 Skip-Gram")):
        _, output, _, evaluation = read_experiment(name)
        score = next(row["ndcg10"] for row in evaluation["summary"] if row["method"] == method)
        assert effect[label].startswith(f"{score:.4f}（")
        with (output / "rankings.csv").open(encoding="utf-8-sig", newline="") as stream:
            expected = [row for row in csv.DictReader(stream)
                        if row["qid"] == query["qid"] and row["method"] == method and int(row["rank"]) <= 10]
        frame = app.dataframe[index].value
        assert frame["新聞 ID"].tolist() == [row["id"] for row in expected]
        assert frame["cosine"].tolist() == pytest.approx([float(row["score"]) for row in expected])
    navigate(app, "詞向量分析")
    assert app.selectbox(key="word_preset").value == app.text_input(key="word_input").value == "earnings"
    assert "earnings 的 Top-5 近鄰" in [element.value for element in app.subheader]
    word_loss_labels = {
        "cbow": ["最高（目前選取）｜300 維 · 窗口 15 · 5 輪｜0.9130", "最低｜100 維 · 窗口 2 · 5 輪｜0.7675"],
        "skipgram": ["最高（目前選取）｜150 維 · 窗口 15 · 15 輪｜0.9474", "最低｜150 維 · 窗口 2 · 5 輪｜0.8395"],
    }
    for method, labels in search_options.items():
        picker = app.selectbox(key=f"word_{method}_run")
        assert picker.options == labels
        if standalone_ready:
            assert picker.value == standalone_best_runs[method]
            high, low = (label.split("｜")[1] for label in word_loss_labels[method])
            assert f"{high} (最高)" in picker.options and f"{low} (最低)" in picker.options
    if standalone_ready:
        assert app.subheader[0].value == "訓練 Loss 比較"
        loss_charts = app.get("vega_lite_chart")[:2]
        assert len(loss_charts) == 2
        for chart, expected in zip(loss_charts, word_loss_labels.values(), strict=True):
            spec = json.loads(chart.proto.spec)
            legend = spec["encoding"]["color"]
            assert legend["title"] == "單用詞向量的平均 nDCG@10"
            assert legend["scale"]["domain"] == expected
    skipgram_neighbors = app.dataframe[1].value.copy()
    app.selectbox(key="word_cbow_run").select(cbow_run).run()
    assert not app.exception and app.dataframe[1].value.equals(skipgram_neighbors)
    cbow_neighbors = app.dataframe[0].value.copy()
    app.selectbox(key="word_skipgram_run").select(skipgram_run).run()
    assert not app.exception and app.dataframe[0].value.equals(cbow_neighbors)
    settings = app.table[0].value
    assert settings.columns.tolist() == ["模型", "向量維度", "窗口（window）", "訓練輪數（epoch）", "最低詞頻（min_count）"]
    assert settings.values.tolist() == [["CBOW", 150, 2, 15, 2], ["Skip-Gram", 150, 5, 15, 2]]
    for index, method, name in ((0, "cbow", cbow_run), (1, "skipgram", skipgram_run)):
        with (ROOT / "artifacts" / "experiments" / name / "word_neighbors.csv").open(encoding="utf-8-sig") as stream:
            expected = [row for row in csv.DictReader(stream) if row["anchor"] == "earnings" and row["model"] == method]
        frame = app.dataframe[index].value
        assert frame["鄰近詞"].tolist() == [row["word"] for row in expected]
        assert frame["cosine"].tolist() == pytest.approx([float(row["cosine"]) for row in expected])
    if standalone_ready:
        for chart, (method, name, window, score) in zip(app.get("vega_lite_chart")[:2],
                                                        (("cbow", cbow_run, 2, "0.8508"),
                                                         ("skipgram", skipgram_run, 5, "0.9196")), strict=True):
            spec = json.loads(chart.proto.spec)
            dataset = next(data for data in chart.proto.datasets if data.name == spec["data"]["name"])
            points = pa.ipc.open_stream(dataset.data.data).read_pandas()
            selected_label = f"目前選取｜150 維 · 窗口 {window} · 15 輪｜{score}"
            selected_line = points.loc[points["series"] == selected_label]
            with (ROOT / "artifacts" / "experiments" / name / method / "epoch_losses.csv").open(encoding="utf-8-sig") as stream:
                saved = list(csv.DictReader(stream))
            assert selected_line["epoch"].tolist() == [int(row["epoch"]) for row in saved]
            assert selected_line["loss_delta"].tolist() == pytest.approx([float(row["loss_delta"]) for row in saved])

        app.selectbox(key="word_cbow_run").select(standalone_best_runs["cbow"]).run()
        assert not app.exception and not app.error
        spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
        assert spec["encoding"]["color"]["scale"]["domain"] == word_loss_labels["cbow"]


def test_experiment_scorecard_rejects_changed_scores_without_loading_model_weights(tmp_path, monkeypatch):
    import os
    from shutil import copyfile
    from ir_hw4 import experiment_ui
    from ir_hw4.experiments import completed_experiments, experiment_paths

    name = experiment_ui.DEFAULT_MODEL_RUN
    if name not in completed_experiments():
        pytest.skip("A completed experiment is required")
    original_run, original_output = experiment_paths(name)
    run, output = tmp_path / "model", tmp_path / "evaluation"
    run.mkdir()
    output.mkdir()
    copyfile(original_run / "manifest.json", run / "manifest.json")
    for filename in ("manifest.json", "ready.json", "results.json"):
        copyfile(original_output / filename, output / filename)
    monkeypatch.setattr(experiment_ui, "experiment_paths", lambda _: (run, output))

    inputs = experiment_ui.scorecard_inputs()
    path = output / "results.json"
    contents = path.read_bytes()
    assert experiment_ui.checked_scorecard(name, *inputs)[2] == json.loads(contents)

    stamp = path.stat()
    field = b'"ndcg10": '
    index = contents.index(field) + len(field)
    path.write_bytes(contents[:index] + b"9" + contents[index + 1:])
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match=r"Fingerprint mismatch: results\.json"):
        experiment_ui.checked_scorecard(name, *inputs)


def test_evaluation_macro_and_single_query(app):
    navigate(app, "效果評估")
    assert len(app.dataframe[0].value) == 3
    assert set(app.dataframe[0].value["查詢數"]) == {"10/10"}
    app.selectbox(key="evaluation_query").select("Q05").run()
    assert not app.exception and len(app.dataframe[1].value) == 30
    app.selectbox(key="evaluation_query").select("全部查詢平均").run()
    assert not app.exception


def test_loss_charts_overlay_saved_extrema_with_their_actual_epoch_lengths(app):
    from ir_hw4.experiment_ui import REFERENCE_RUNS
    from ir_hw4.experiments import completed_experiments

    if not set(REFERENCE_RUNS).issubset(completed_experiments()):
        pytest.skip("All reference runs are required to compare the extrema")
    navigate(app, "效果評估")
    app.radio(key="evaluation_mode").set_value("參數實驗").run()
    options = app.multiselect(key="comparison_experiments").options
    assert set(REFERENCE_RUNS) <= set(options)
    assert not {"正式模型", "grid-d100-w2", "grid-d150-w2", "grid-d150-w5"}.intersection(options)
    app.radio(key="parameter_experiment_view").set_value("訓練 Loss 曲線").run()
    assert not app.exception and not app.error
    assert app.radio(key="experiment_loss_view").value == "最高與最低"
    assert not app.selectbox
    charts = app.get("vega_lite_chart")
    assert len(charts) == 2
    expected = {
        "cbow": [("最高｜100 維 · 窗口 15 · 5 輪｜0.9206", "grid-d100-w15-e5"),
                 ("最低｜150 維 · 窗口 2 · 5 輪｜0.7958", "grid-d150-w2-e5")],
        "skipgram": [("最高｜100 維 · 窗口 15 · 15 輪｜0.9526", "grid-d100-w15-e15"),
                     ("最低｜300 維 · 窗口 2 · 5 輪｜0.8148", "grid-d300-w2-e5")],
    }
    for chart, (method, runs) in zip(charts, expected.items()):
        spec = json.loads(chart.proto.spec)
        assert spec["encoding"]["color"]["scale"]["domain"] == [label for label, _ in runs]
        dataset = next(data for data in chart.proto.datasets if data.name == spec["data"]["name"])
        points = pa.ipc.open_stream(dataset.data.data).read_pandas()
        for label, name in runs:
            with (ROOT / "artifacts" / "experiments" / name / method / "epoch_losses.csv").open(encoding="utf-8-sig") as stream:
                saved = list(csv.DictReader(stream))
            line = points[points["series"] == label]
            assert line["epoch"].tolist() == [int(row["epoch"]) for row in saved]
            assert line["loss_delta"].tolist() == pytest.approx([float(row["loss_delta"]) for row in saved])
    app.radio(key="experiment_loss_view").set_value("單組實驗").run()
    assert not app.exception and len(app.get("vega_lite_chart")) == 2
    assert all("30 輪" not in label for label in app.selectbox(key="experiment_loss_run").options)


def test_ui_submits_independent_settings_only_on_explicit_training_click(app, monkeypatch):
    import ir_hw4.training
    import ir_hw4.dataset
    import ir_hw4.experiments
    from ir_hw4.training_jobs import TrainingJobs

    def forbidden(*args, **kwargs):
        raise AssertionError("The website may only submit new experiments through its training form")

    monkeypatch.setattr(ir_hw4.training, "train", forbidden)
    monkeypatch.setattr(ir_hw4.dataset, "prepare_data", forbidden)
    monkeypatch.setattr(ir_hw4.experiments, "train_experiment", forbidden)
    submitted = []

    def start(self, name, settings):
        submitted.append((name, settings))
        return name

    monkeypatch.setattr(TrainingJobs, "latest", lambda self: None)
    monkeypatch.setattr(TrainingJobs, "start", start)
    for page in ("詞向量分析", "新聞搜尋與比較", "效果評估", "資料概況"):
        navigate(app, page)
    navigate(app, "效果評估")
    app.radio(key="evaluation_mode").set_value("參數實驗").run()
    app.radio(key="parameter_experiment_view").set_value("自行訓練").run()
    assert not app.exception and submitted == []
    app.selectbox(key="training_reference").select((300, 15, 15)).run()
    next(button for button in app.button if button.label == "填入兩欄").click().run()
    for model in ("cbow", "skipgram"):
        actual = tuple(app.number_input(key=f"experiment_{model}_{field}").value
                       for field in ("vector_size", "window", "epochs"))
        assert actual == (300, 15, 15)
    assert submitted == []
    app.text_input(key="new_experiment_name").set_value("example-independent")
    app.number_input(key="experiment_cbow_vector_size").set_value(128)
    app.number_input(key="experiment_cbow_window").set_value(7)
    app.number_input(key="experiment_cbow_epochs").set_value(5)
    app.number_input(key="experiment_skipgram_vector_size").set_value(150)
    app.number_input(key="experiment_skipgram_window").set_value(3)
    next(button for button in app.button if button.label == "開始訓練").click().run()
    assert not app.exception and len(submitted) == 1
    name, settings = submitted[0]
    assert name == "example-independent"
    assert (settings["cbow"]["vector_size"], settings["cbow"]["window"], settings["cbow"]["epochs"]) == (128, 7, 5)
    assert (settings["skipgram"]["vector_size"], settings["skipgram"]["window"], settings["skipgram"]["epochs"]) == (150, 3, 15)
