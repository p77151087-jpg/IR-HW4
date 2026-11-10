"""Configure, follow, and compare local Word2Vec experiments."""

from hashlib import sha256 as hash_bytes
from io import BytesIO
from itertools import product
import json
from math import isfinite

import altair as alt
import pandas as pd
import streamlit as st

from .analysis import WORD_MODELS, checked_evaluation
from .config import DATA, METHODS, ROOT, RUN
from .evaluation import query_bank
from .experiments import (
    DEFAULT_CONFIG, EXPERIMENT_SOURCE, PARAMETERS, check_hashes, completed_experiments,
    experiment_paths, read_experiment,
)
from .io import read_json, sha256
from .training import PRODUCER_FILES
from .training_jobs import ACTIVE_PHASES, MIN_WEB_VECTOR_SIZE, TrainingJobs

REFERENCE_SETTINGS = tuple(product((100, 150, 300), (2, 5, 15), (5, 15)))
REFERENCE_RUNS = tuple(f"grid-d{size}-w{window}-e{epochs}" for size, window, epochs in REFERENCE_SETTINGS)
DEFAULT_REFERENCE = (100, 5, 5)
DEFAULT_MODEL_RUN = "grid-d100-w5-e15"
HIDDEN_EPOCHS = 30
PRIMARY_FIELDS = ("vector_size", "window", "min_count", "epochs")
SCORE_SOURCES = {path for path in PRODUCER_FILES if path.endswith(".py")} | {"ir_hw4/evaluation.py"}


def model_setting_label(config):
    return f"v{config['vector_size']}-w{config['window']}-e{config['epochs']}"


def model_setting_description(config):
    return f"{config['vector_size']} 維 · 窗口 {config['window']} · {config['epochs']} 輪"


def checked_bytes(directory, relative, digest):
    root = directory.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Experiment file escapes its directory: {relative}")
    payload = path.read_bytes()
    if hash_bytes(payload).hexdigest() != digest:
        raise ValueError(f"Fingerprint mismatch: {relative}")
    return payload


def completed_manifests(name):
    run, output = experiment_paths(name)
    ready = read_json(output / "ready.json")
    if not isinstance(ready, dict) or ready.keys() != {"model_manifest_sha256", "evaluation_manifest_sha256"}:
        raise ValueError("Experiment completion record does not match its manifests")
    model_signature = ready["model_manifest_sha256"]
    model = json.loads(checked_bytes(run, "manifest.json", model_signature))
    evaluation = json.loads(checked_bytes(output, "manifest.json", ready["evaluation_manifest_sha256"]))
    if model["status"] != "complete" or model["experiment_name"] != name or evaluation["status"] != "complete":
        raise ValueError("Experiment is incomplete")
    if evaluation["model_manifest_sha256"] != model_signature:
        raise ValueError("Experiment evaluation belongs to a different model")
    return run, output, model, evaluation


def scorecard_inputs():
    """Hash the shared experiment inputs once for a whole comparison."""
    return {"base_model_manifest_sha256": sha256(RUN / "manifest.json"),
            "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
            "corpus_sha256": sha256(DATA / "documents.jsonl")}, sha256(ROOT / "resources" / "queries.json")


def checked_scorecard(name, model_inputs, query_signature):
    """Verify the saved scores without reading model weights or exported rankings."""
    run, output, model, evaluation = completed_manifests(name)
    if any(model[key] != digest for key, digest in model_inputs.items()):
        raise ValueError("Experiment data or reference model does not match")
    if (evaluation["queries_sha256"] != query_signature or evaluation["test_documents"] != model["test_documents"]
            or evaluation["methods"] != METHODS or evaluation["k"] != 10 or evaluation["source"] != "reuters_topics"):
        raise ValueError("Experiment evaluation does not match the current comparison protocol")
    check_hashes(ROOT, model["source_sha256"], {*PRODUCER_FILES, EXPERIMENT_SOURCE})
    check_hashes(ROOT, {path: evaluation["source_sha256"][path] for path in SCORE_SOURCES}, SCORE_SOURCES)
    payload = checked_bytes(output, "results.json", evaluation["artifacts_sha256"]["results.json"])
    return run, model, json.loads(payload)


def website_experiment_configs():
    """List eligible model settings from completed manifests for website selectors."""
    experiments = {}
    for name in completed_experiments():
        try:
            _, _, manifest, _ = completed_manifests(name)
            configs = {method: manifest["models"][method]["config"] for method in WORD_MODELS}
            visible = {method: config for method, config in configs.items()
                       if config["vector_size"] >= MIN_WEB_VECTOR_SIZE and config["epochs"] != HIDDEN_EPOCHS}
            if visible:
                experiments[name] = visible
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return experiments


def score_extreme_marks(scores):
    rounded = {name: round(score, 12) for name, score in scores.items() if score is not None}
    if len(rounded) < 2:
        return {}
    values = list(rounded.values())
    low, high = min(values), max(values)
    if low == high:
        return dict.fromkeys(rounded, "同分")
    highest = "並列最高" if values.count(high) > 1 else "最高"
    lowest = "並列最低" if values.count(low) > 1 else "最低"
    return {name: highest if score == high else lowest
            for name, score in rounded.items() if score in (low, high)}


def reference_model_extremes(options, *, ndcg10_by_model=None):
    """Return the verified reference-grid extrema for model controls and Loss charts."""
    if not all(name in options and options[name].keys() == WORD_MODELS.keys() for name in REFERENCE_RUNS):
        return {}
    extremes = {}
    try:
        inputs = scorecard_inputs()
        runs = {name: checked_scorecard(name, *inputs) for name in REFERENCE_RUNS}
        if ndcg10_by_model is None:
            ndcg10_by_model = {(name, row["method"]): row["ndcg10"]
                               for name, (_, _, result) in runs.items() for row in result["summary"]}
        for method in WORD_MODELS:
            scores = {name: ndcg10_by_model[name, method] for name in REFERENCE_RUNS}
            if all(score is not None for score in scores.values()):
                for name, mark in score_extreme_marks(scores).items():
                    run, manifest, _ = runs[name]
                    extremes[name, method] = (mark, run, manifest["models"][method]["config"],
                                              manifest["artifacts_sha256"][f"{method}/epoch_losses.csv"], scores[name])
    except (ValueError, OSError, KeyError, TypeError):
        return {}
    return extremes


@st.cache_resource
def training_jobs():
    return TrainingJobs()


def input_parameter(model, field):
    minimum, maximum, description = PARAMETERS[field]
    if field == "vector_size":
        minimum = MIN_WEB_VECTOR_SIZE
    default = DEFAULT_REFERENCE[2] if field == "epochs" else getattr(DEFAULT_CONFIG, field)
    options = {"step": .001, "format": "%.6f"} if isinstance(default, float) else {"step": 1}
    key = f"experiment_{model}_{field}"
    value = None if key in st.session_state else default
    return st.number_input(field, min_value=minimum, max_value=maximum, value=value, help=description, key=key, **options)


def apply_reference():
    dimension, window, epochs = st.session_state["training_reference"]
    overrides = {"vector_size": dimension, "window": window, "epochs": epochs}
    for model in WORD_MODELS:
        for field in PARAMETERS:
            st.session_state[f"experiment_{model}_{field}"] = overrides.get(field, getattr(DEFAULT_CONFIG, field))


def compare_job(name):
    st.session_state["submitted_training"] = name
    st.session_state["parameter_experiment_view"] = "比較結果"


def training_status(job):
    phase = job["phase"]
    name = job["name"]
    if phase == "complete":
        st.success(f"{name} 已完成訓練與十個查詢的評估。")
        st.button("比較這次結果", on_click=compare_job, args=(name,))
        st.caption("也可到「詞向量分析」選取這組模型。")
    elif phase == "failed":
        st.error(f"{name} 的訓練未完成。{job['error']}")
        st.caption(f"詳細記錄位於 {ROOT / 'reports' / 'training-jobs' / name / 'worker.log'}。")
    elif phase in ("training", "saving"):
        model = WORD_MODELS[job["model"]]
        detail = " · 正在保存模型與測試文件向量" if phase == "saving" else ""
        st.progress(job["epoch"] / job["epochs"], text=f"{name} · {model} · epoch {job['epoch']}/{job['epochs']}{detail}")
    else:
        detail = {"queued": "啟動本機訓練", "preparing": "核對訓練語料與既有權重",
                  "evaluating": "評估固定十個查詢"}[phase]
        st.info(f"{name} · 正在{detail}…")


@st.fragment(run_every="1s")
def follow_training():
    job = training_jobs().latest()
    if job is None or job["phase"] not in ACTIVE_PHASES:
        st.rerun()
    training_status(job)


def training_controls():
    manager = training_jobs()
    job = manager.latest()
    busy = job is not None and job["phase"] in ACTIVE_PHASES
    st.subheader("分別設定 CBOW 與 Skip-Gram")
    st.caption(f"兩欄分開設定，向量維度從 {MIN_WEB_VECTOR_SIZE} 起。送出後，本機依序訓練兩個模型，再用固定十個查詢評估。")
    a, b = st.columns([2, 1], vertical_alignment="bottom")
    a.selectbox("參考組合", REFERENCE_SETTINGS,
                index=REFERENCE_SETTINGS.index(DEFAULT_REFERENCE), key="training_reference",
                format_func=lambda setting: f"{setting[0]} 維 · window {setting[1]} · {setting[2]} epochs")
    b.button("填入兩欄", on_click=apply_reference)
    with st.form("parameter_training"):
        name = st.text_input("實驗名稱", key="new_experiment_name", placeholder="留空會自動命名",
                             help="可用英數、連字號與底線。每次訓練使用新名稱。")
        settings = {}
        for col, (model, label) in zip(st.columns(2, gap="large"), WORD_MODELS.items()):
            with col:
                st.markdown(f"#### {label}")
                settings[model] = {field: input_parameter(model, field) for field in PRIMARY_FIELDS}
                with st.expander(f"{label} 進階參數"):
                    settings[model].update({field: input_parameter(model, field) for field in PARAMETERS
                                            if field not in PRIMARY_FIELDS})
        submitted = st.form_submit_button("開始訓練", type="primary", disabled=busy)
    if submitted:
        try:
            if any(config["epochs"] == HIDDEN_EPOCHS for config in settings.values()):
                raise ValueError("網站已移除 30 輪的組合。請調整訓練輪數後再送出。")
            st.session_state["submitted_training"] = manager.start(name, settings)
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.rerun()
    if busy:
        follow_training()
    elif job is not None:
        if job["phase"] != "complete" or website_experiment_configs().get(job["name"], {}).keys() == WORD_MODELS.keys():
            training_status(job)


def comparison_panel(signature):
    st.subheader("比較訓練效果")
    st.caption("18 組參考實驗使用 100／150／300 維 × window 2／5／15 × 5／15 epochs。")
    available = [name for name, configs in website_experiment_configs().items() if configs.keys() == WORD_MODELS.keys()]
    reference = [name for name in REFERENCE_RUNS if name in available]
    if len(reference) < len(REFERENCE_RUNS):
        st.info(f"參考實驗已完成 {len(reference)}／{len(REFERENCE_RUNS)} 組，目前只比較已完成的組合。")
    if not available:
        return
    baseline, _ = checked_evaluation(signature)
    names = [*reference, *(name for name in available if name not in reference)]
    defaults = reference or names[:1]
    submitted = st.session_state.get("submitted_training")
    if submitted in available and submitted not in defaults:
        defaults = [*defaults, submitted]
    with st.expander("調整要比較的實驗"):
        selected = st.multiselect("已完成的實驗", names, default=defaults, key="comparison_experiments")
    runs = []
    inputs = scorecard_inputs() if selected else None
    for name in selected:
        try:
            _, manifest, results = checked_scorecard(name, *inputs)
        except (ValueError, OSError, KeyError, TypeError) as error:
            st.error(f"{name} 的資料或實驗驗證失敗。{error}")
            continue
        runs.append((name, manifest, results))
    if not runs:
        st.info("請選擇至少一組已完成的實驗。")
        return
    queries = {query["qid"]: query["text"] for query in query_bank()}
    a, b = st.columns(2)
    query = a.selectbox("比較範圍", ["全部查詢平均", *queries], key="experiment_comparison_query",
                        format_func=lambda key: f"{key} · {queries[key]}" if key in queries else key)
    mean = query == "全部查詢平均"
    labels = {"p10": "P@10", "mrr": "MRR" if mean else "RR", "ndcg10": "nDCG@10"}
    criterion = b.selectbox("最高／最低依據", [labels["ndcg10"], labels["p10"], labels["mrr"]],
                            key="experiment_extreme_metric")
    original = {row["method"]: row for row in (baseline["summary"] if mean else baseline["per_query"])
                if mean or row["qid"] == query}
    table = []
    for name, manifest, results in runs:
        for row in (results["summary"] if mean else results["per_query"]):
            method = row["method"]
            if method not in WORD_MODELS or (not mean and row["qid"] != query):
                continue
            config = manifest["models"][method]["config"]
            item = {"實驗": name, "模型": WORD_MODELS[method], "vector_size": config["vector_size"],
                    "window": config["window"], "epochs": config["epochs"],
                    **{label: row[key] for key, label in labels.items()}, "組內比較": ""}
            for key, label in labels.items():
                reference = original[method][key]
                item[f"Δ {label}"] = row[key] - reference if reference is not None and row[key] is not None else None
            if not mean:
                item["有效候選"] = row["valid_documents"]
                item["詞向量權重涵蓋率"] = row["embedding_weight_coverage"]
            table.append(item)
    for label in WORD_MODELS.values():
        model_rows = {row["實驗"]: row for row in table if row["模型"] == label}
        marks = score_extreme_marks({name: row[criterion] for name, row in model_rows.items()})
        for name, mark in marks.items():
            model_rows[name]["組內比較"] = mark
    frame = pd.DataFrame(table)
    marks = frame.pop("組內比較")
    frame.insert(frame.columns.get_loc(criterion) + 1, "組內比較", marks)
    column_config = {column: st.column_config.NumberColumn(format="%.4f")
                     for label in labels.values() for column in (label, f"Δ {label}")}
    column_config["組內比較"] = st.column_config.TextColumn("最高／最低", width="small")
    extremes = frame.loc[frame["組內比較"].isin(("最高", "最低", "並列最高", "並列最低")),
                         ["模型", "實驗", "vector_size", "window", "epochs", criterion, "組內比較"]]
    if not extremes.empty:
        st.markdown("**各模型的最高與最低**")
        st.dataframe(extremes.sort_values(["模型", criterion], ascending=[True, False]),
                     hide_index=True, width="stretch", column_config=column_config)
    st.markdown("**各組完整指標**")
    st.dataframe(frame, hide_index=True, width="stretch", column_config=column_config)
    st.caption(f"最高與最低依 {criterion} 在各架構內分別比較目前表中的實驗。Δ 以同架構的正式模型為基準。"
               "各組使用相同的 ModApte 切分、TF-IDF 權重、查詢與 topics 評估規則。")
    with st.expander("查看兩種模型的完整設定"):
        settings = [{"實驗": name, "模型": label, **{field: manifest["models"][model]["config"][field] for field in PARAMETERS}}
                    for name, manifest, _ in runs for model, label in WORD_MODELS.items()]
        st.dataframe(pd.DataFrame(settings), hide_index=True, width="stretch")


def epoch_losses(run, model, config, mark="", digest=None, ndcg10=None):
    relative = f"{model}/epoch_losses.csv"
    source = run / relative if digest is None else BytesIO(checked_bytes(run, relative, digest))
    frame = pd.read_csv(source, usecols=["epoch", "loss_delta"])
    if frame["epoch"].tolist() != list(range(1, config["epochs"] + 1)) or not all(map(isfinite, frame["loss_delta"])):
        raise ValueError(f"{WORD_MODELS[model]} 的每輪 Loss 紀錄不完整。")
    settings = model_setting_description(config)
    if mark:
        score = f"{ndcg10:.4f}" if ndcg10 is not None else "—"
        settings = f"{mark}｜{settings}｜{score}"
    return frame.assign(series=settings)


def loss_chart(frame, legend_title=None):
    chart = alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="訓練輪數（Epoch）", scale=alt.Scale(zero=False), axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("loss_delta:Q", title="每輪訓練 Loss", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("series:N", title="參數組合"), alt.Tooltip("epoch:Q", title="Epoch", format="d"),
                 alt.Tooltip("loss_delta:Q", title="每輪 Loss", format=",.2f")],
    )
    labels = frame["series"].drop_duplicates().tolist()
    if len(labels) > 1 or legend_title is not None:
        colors = ["#176E65", "#B75228", "#2455AD"]
        dashes = [[1, 0], [6, 4], [2, 3]]
        legend = alt.Legend(orient="bottom", direction="vertical", labelLimit=0)
        if legend_title is not None:
            # Show the score with its role, then put the parameters on a second line.
            legend.labelExpr = "[split(datum.label, '｜')[0] + ' · ' + split(datum.label, '｜')[2], split(datum.label, '｜')[1]]"
            legend.rowPadding = 7
        chart = chart.encode(
            color=alt.Color("series:N", title=legend_title, sort=labels,
                            scale=alt.Scale(domain=labels, range=colors[:len(labels)]),
                            legend=legend),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=labels, range=dashes[:len(labels)]), legend=None),
        )
    return chart.properties(height=320 if legend_title is not None else 275)


def show_run_losses(run, settings):
    for col, (model, label) in zip(st.columns(2, gap="large"), WORD_MODELS.items()):
        col.markdown(f"#### {label}")
        config = settings[model]
        col.caption(model_setting_description(config))
        try:
            losses = epoch_losses(run, model, config)
        except (ValueError, OSError, KeyError, TypeError) as error:
            col.error(f"{run.name} 的 Loss 無法顯示。{error}")
            continue
        col.altair_chart(loss_chart(losses), width="stretch")


def compare_extreme_losses(extremes, selected=None, *, score_label="平均 nDCG@10"):
    st.caption(f"最高與最低依 18 組參考實驗在固定十個查詢上的{score_label}，各架構分開排名。"
               "同張圖的曲線共用縱軸，各自停在最後一輪。")
    selected = selected or {}
    for col, (model, label) in zip(st.columns(2, gap="large"), WORD_MODELS.items()):
        col.markdown(f"#### {label}")
        candidates = [item for (_, method), item in extremes.items() if method == model]
        highest = next((item for item in candidates if item[0].endswith("最高")), None)
        lowest = next((item for item in candidates if item[0].endswith("最低")), None)
        if highest is None or lowest is None:
            col.info("各參考組合的評估結果需通過驗證，且平均 nDCG@10 有高低差，才能顯示最高與最低曲線。")
        series_by_run = {item[1]: item for item in (highest, lowest) if item is not None}
        if model in selected:
            run, config, digest, score = selected[model]
            mark = f"{series_by_run[run][0]}（目前選取）" if run in series_by_run else "目前選取"
            series_by_run[run] = (mark, run, config, digest, score)
        if not series_by_run:
            continue
        try:
            frame = pd.concat([epoch_losses(run, model, config, mark, digest, score)
                               for mark, run, config, digest, score in series_by_run.values()],
                              ignore_index=True)
        except (ValueError, OSError, KeyError, TypeError) as error:
            col.error(f"{label} 的 Loss 無法顯示。{error}")
            continue
        col.altair_chart(loss_chart(frame, legend_title=score_label), width="stretch")


def loss_panel():
    st.subheader("訓練 Loss 曲線")
    configs = {name: models for name, models in website_experiment_configs().items()
               if models.keys() == WORD_MODELS.keys()}
    if not configs:
        st.info("完成訓練後可查看每輪 Loss。")
        return
    view = st.radio("Loss 圖表", ("最高與最低", "單組實驗"), horizontal=True, key="experiment_loss_view")
    st.caption("縱軸是每輪新增的訓練損失。兩種架構各用自己的縱軸刻度。")
    if view == "最高與最低":
        compare_extreme_losses(reference_model_extremes(configs))
        return
    names = [name for name in REFERENCE_RUNS if name in configs]
    names.extend(name for name in configs if name not in REFERENCE_RUNS)
    name = st.selectbox("選擇實驗", names, key="experiment_loss_run",
                        format_func=lambda name: model_setting_description(configs[name]["cbow"])
                        if name in REFERENCE_RUNS else name)
    try:
        run, _, manifest, _ = read_experiment(name)
        settings = {model: manifest["models"][model]["config"] for model in WORD_MODELS}
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f"{name} 的 Loss 無法顯示。{error}")
        return
    show_run_losses(run, settings)


def experiment_panel(signature):
    view = st.radio("參數實驗內容", ("比較結果", "訓練 Loss 曲線", "自行訓練"), horizontal=True, key="parameter_experiment_view")
    if view == "自行訓練":
        training_controls()
    elif view == "訓練 Loss 曲線":
        loss_panel()
    else:
        comparison_panel(signature)
