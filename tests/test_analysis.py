from pathlib import Path
import shutil
import zipfile
import pytest
from streamlit.testing.v1 import AppTest

from ir_hw4 import analysis
from ir_hw4.config import ROOT, ANALYSIS_WORDS


@pytest.fixture(scope="module")
def report():
    return analysis.load_analysis()


def test_every_fixed_word_has_both_top5_three_explanations_and_train_evidence(report):
    payload, _ = report
    assert [entry["word"] for entry in payload["words"]] == ANALYSIS_WORDS
    for entry in payload["words"]:
        assert all(entry[key].strip() for key in ("reasonable", "context", "unexpected"))
        assert all(len(entry["neighbors"][model]) == 5 for model in ("cbow", "skipgram"))
        unique_neighbors = {row["word"] for rows in entry["neighbors"].values() for row in rows}
        assert set(entry["annotations"]) == unique_neighbors
        assert {row["neighbor"] for row in entry["evidence"]} == unique_neighbors
        assert all(row["witnesses"] for row in entry["evidence"])
    acquisition = next(entry for entry in payload["words"] if entry["word"] == "acquisition")
    sahlen = next(row for row in acquisition["evidence"] if row["neighbor"] == "sahlen")
    assert sahlen["cooccurrence_documents"] == 0
    assert len({w["id"] for w in sahlen["witnesses"]}) == 2


def test_cases_keep_improvement_failure_and_coarse_label_distinct(report):
    payload, _ = report
    cases = {case["query"]["qid"]: case for case in payload["cases"]}
    improvement = {row["method"]: row for row in cases["Q06"]["metrics"]}
    assert improvement["skipgram"]["p10"] == 1 and improvement["tfidf"]["p10"] == .4
    assert improvement["cbow"]["mrr"] < improvement["tfidf"]["mrr"]
    failure = {row["method"]: row for row in cases["Q09"]["metrics"]}
    assert {row["p10"] for row in failure.values()} == {.8}
    assert failure["skipgram"]["ndcg10"] < failure["tfidf"]["ndcg10"]
    assert failure["cbow"]["mrr"] == .5
    brief = next(row for row in cases["Q09"]["evidence"] if row["id"] == "20082")
    assert brief["title_only"] and "\n" not in brief["text"]
    loss = next(row for row in cases["Q01"]["documents"] if row["id"] == "16711")
    assert loss["topic_relevant"] and loss["ranks"]["cbow"] == 2
    assert "不變更查詢" in cases["Q01"]["limitation"]


def test_integrated_report_and_zip_include_word_document_levels_and_data(report):
    _, directory = report
    content = (directory / "HW4_REPORT.md").read_text(encoding="utf-8")
    for word in ANALYSIS_WORDS:
        assert f"### {word}\n" in content
    for heading in ("合理關係", "新聞情境", "意外結果與可能原因", "Q06", "Q09", "Q01",
                    "逐查詢結果", "負 cosine", "negative", "重現與來源"):
        assert heading in content
    assert content.count("**合理關係**") == 10
    with zipfile.ZipFile(directory / "HW4_REPORT.zip") as package:
        assert {"HW4_REPORT.md", "analysis.json", "word_neighbors.csv", "metrics_per_query.csv",
                "metrics_summary.csv", "queries.json", "sources.json"}.issubset(package.namelist())
        assert package.read("HW4_REPORT.md") == (directory / "HW4_REPORT.md").read_bytes()


def test_word_evidence_tables_are_complete_before_source_excerpts(report):
    _, directory = report
    content = (directory / "HW4_REPORT.md").read_text(encoding="utf-8")
    sections = content.split("| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |\n")[1:]
    assert len(sections) == 10
    for section in sections:
        table_lines = section.split("\n\n", 1)[0].splitlines()
        # Delimiter plus two evidence rows; neither row is detached below the quotations.
        assert len(table_lines) == 3


def test_corrupted_integrated_report_is_rejected(report, tmp_path):
    _, directory = report
    target = tmp_path / "copy"
    shutil.copytree(directory, target)
    with (target / "HW4_REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\ncorrupted")
    with pytest.raises(ValueError, match="Analysis artifact mismatch"):
        analysis.load_analysis(target)


def test_ui_only_change_can_reuse_scores_but_scoring_change_is_rejected(monkeypatch):
    original = analysis.sha256
    def changed_ui(path):
        return "display-only-edit" if Path(path) == ROOT / "ir_hw4/ui.py" else original(path)
    monkeypatch.setattr(analysis, "sha256", changed_ui)
    result, _ = analysis.checked_evaluation()
    assert len(result["summary"]) == 3
    def changed_scoring(path):
        return "changed" if Path(path) == ROOT / "ir_hw4/retrieval.py" else original(path)
    monkeypatch.setattr(analysis, "sha256", changed_scoring)
    with pytest.raises(ValueError, match="Evaluation source changed"):
        analysis.checked_evaluation()


def test_word_interpretation_follows_submitted_word_and_works_for_all_ten():
    from ir_hw4.experiment_ui import DEFAULT_MODEL_RUN
    from ir_hw4.experiments import completed_experiments

    if DEFAULT_MODEL_RUN not in completed_experiments():
        pytest.skip("The 100-dimension, window-5 run is required for these word examples")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.sidebar.radio(key="page").set_value("詞向量分析").run()
    at.selectbox(key="word_cbow_run").select(DEFAULT_MODEL_RUN)
    at.selectbox(key="word_skipgram_run").select(DEFAULT_MODEL_RUN).run()
    for word in ANALYSIS_WORDS:
        at.selectbox(key="word_preset").select(word).run()
        at.button[0].click().run(timeout=90)
        assert not at.exception
        assert f"{word} 的逐詞解讀" in [element.value for element in at.subheader]
        assert f"{word} 的近鄰詞解讀" in [element.value for element in at.subheader]
        text = "\n".join(element.value for element in at.markdown)
        assert "合理的語意或使用情境關係" in text and "相似新聞情境" in text and "可能原因為何" in text
        assert any("各圖只顯示本模型的 Top-5" in element.value for element in at.caption)
    at.selectbox(key="word_preset").select("oil").run()
    at.button[0].click().run()
    at.selectbox(key=f"neighbor_detail_oil_{DEFAULT_MODEL_RUN}").select("kernel").run()
    assert not at.exception
    assert any("共現 5 篇" in element.value for element in at.caption)
    assert any("訓練新聞 885" in element.value for element in at.markdown)


def test_word_analysis_covers_new_neighbors_with_real_training_evidence():
    from ir_hw4.experiments import completed_experiments
    from ir_hw4.text import terms

    run = "grid-d100-w15-e15"
    if run not in completed_experiments():
        pytest.skip("The 100-dimension, window-15 run is required for these neighbor examples")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.sidebar.radio(key="page").set_value("詞向量分析").run()
    at.selectbox(key="word_cbow_run").select(run)
    at.selectbox(key="word_skipgram_run").select(run).run()
    at.selectbox(key="word_preset").select("oil").run()
    at.button[0].click().run()
    assert not at.exception
    energy = at.dataframe[0].value.set_index("鄰近詞").loc["energy"]
    assert energy["名次"] == 3 and energy["cosine"] == pytest.approx(0.5755577468395261)
    assert energy["語境判讀"] == "能源與油市" and energy["同篇共現（篇）"] == 149
    assert at.dataframe[0].value.set_index("鄰近詞").loc["crude", "語境判讀"] == "原油搭配"
    at.selectbox(key=f"neighbor_detail_oil_{run}").select("energy").run()
    assert not at.exception
    assert any("判讀依據 · 補充逐詞解讀" in item.value for item in at.caption)
    assert any("訓練新聞 144" in item.value for item in at.markdown)
    assert any({"oil", "energy"} <= set(terms(item.value)) for item in at.text)

    at.text_input(key="word_input").set_value("aluminum")
    at.button[0].click().run()
    assert not at.exception
    for table in at.dataframe[:2]:
        assert table.value["同篇共現（篇）"].notna().all()
        assert not table.value["語境判讀"].str.contains("曾在同一句話出現").any()
    polyethylene = at.dataframe[0].value.set_index("鄰近詞").loc["polyethylene"]
    assert polyethylene["語境判讀"] == "各自的周邊用詞" and polyethylene["同篇共現（篇）"] == 0
    at.selectbox(key=f"neighbor_detail_aluminum_{run}").select("polyethylene").run()
    assert not at.exception
    assert any("smelter" in item.value and "density" in item.value for item in at.markdown)
    evidence = next(expander for expander in at.expander if expander.label == "原文證據 · Reuters 訓練新聞")
    excerpts = [set(terms(item.value)) for item in evidence.text]
    assert len(excerpts) == 2
    assert any("aluminum" in excerpt for excerpt in excerpts)
    assert any("polyethylene" in excerpt for excerpt in excerpts)

    at.text_input(key="word_input").set_value("earnings")
    at.button[0].click().run()
    assert not at.exception
    rendered = [item.value for item in at.markdown]
    first = rendered.index("**1. 哪些鄰近詞具有合理的語意或使用情境關係？**")
    second = rendered.index("**2. 哪些結果可能來自經常出現的相似新聞情境？**")
    third = rendered.index("**3. 哪些結果不符合預期？可能原因為何？**")
    assert any("**和 `earnings` 的關係**" in item and "net income" in item for item in rendered[first:second])
    paired = rendered[rendered.index("#### `re-classification`", second):third]
    assert any("貸款重新分類與銀行盈餘" in item for item in paired)
    assert "**共同用詞 `third`**" in paired and "**共同用詞 `expects`**" in paired
    assert "**在 `earnings` 旁邊** · 11 篇訓練新聞" in paired
    assert "**在 `earnings` 旁邊** · 48 篇訓練新聞" in paired
    assert paired.count("**在 `re-classification` 旁邊** · 2 篇訓練新聞") == 2
    original_examples = [" ".join(item.value.split()) for item in at.html]
    assert any("<mark>earnings</mark>" in item and "<mark>income</mark>" in item for item in original_examples)
    assert any("<mark>earnings</mark> for the <mark>third</mark> quarter" in item for item in original_examples)
    assert any("<mark>re-classification</mark> of as much as a <mark>third</mark> of the loans" in item
               for item in original_examples)
    assert any("<mark>expects</mark> <mark>earnings</mark>" in item for item in original_examples)
    assert any("<mark>expects</mark> <mark>re-classification</mark>" in item for item in original_examples)

    examples = (("rentcorp", "McGrath RentCorp", 2560, 8), ("thrid", "拼字錯誤", 10096, 3),
                ("graff", "Joel Graff", 2729, 10), ("telxon", "公司名稱 Telxon", 1339, 5))
    for term, meaning, news, distance in examples:
        details = rendered[rendered.index(f"#### `{term}`") + 1]
        assert meaning in details and "為什麼排進前五名會讓人意外" in details
        assert f"相距 {distance} 個詞位" in details and "Skip-Gram 的窗口 15" in details
        assert f"**原文例句 · 訓練新聞 #{news}**" in rendered
        assert any({term, "earnings"} <= set(terms(item.value)) for item in at.text)


def test_case_switching_and_report_downloads_preserve_four_pages():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert len(at.sidebar.radio(key="page").options) == 4
    at.sidebar.radio(key="page").set_value("效果評估").run(timeout=90)
    for qid in ("Q09", "Q01", "Q06"):
        at.selectbox(key="case_qid").select(qid).run(timeout=90)
        assert not at.exception
        assert any(qid in e for e in at.selectbox(key="case_qid").options)
        assert at.selectbox(key="case_qid").value == qid
        assert any("可能原因" in e.value for e in at.markdown)
    at.selectbox(key="evaluation_query").select("Q09").run()
    assert at.selectbox(key="case_qid").value == "Q09"
    assert len(at.sidebar.radio(key="page").options) == 4
