from collections import Counter
import math
import numpy as np
import pytest

from ir_hw4.dataset import ReutersParser
from ir_hw4.text import terms, document_text
from ir_hw4.retrieval import TfidfSpace, weighted_documents, RetrievalEngine, neighbors, ordered_indices
from ir_hw4.evaluation import metrics


def test_modapte_uses_flag_not_nonempty_topics_and_preserves_multilabel():
    parser = ReutersParser()
    parser.feed('''<REUTERS NEWID="1" LEWISSPLIT="TRAIN" TOPICS="YES"><TOPICS></TOPICS><TEXT><TITLE>Oil &amp; gas</TITLE><BODY>Petroleum.</BODY></TEXT></REUTERS>
<REUTERS NEWID="2" LEWISSPLIT="TEST" TOPICS="NO"><TOPICS><D>crude</D></TOPICS><TEXT TYPE="BRIEF"><TITLE>Oil</TITLE></TEXT></REUTERS>
<REUTERS NEWID="3" LEWISSPLIT="TEST" TOPICS="YES"><TOPICS><D>crude</D><D>trade</D></TOPICS><TEXT TYPE="UNPROC">Raw news</TEXT></REUTERS>''')
    a, b, c = parser.documents
    assert a["split"] == "train" and a["topics"] == [] and a["title"] == "Oil & gas"
    assert b["split"] == "unused" and document_text(b) == "Oil"
    assert c["split"] == "test" and c["topics"] == ["crude", "trade"] and c["body"] == "Raw news"


def test_shared_preprocessing_preserves_query_tf_and_negation():
    assert terms("The OIL oil, NOT petroleum; wheat’s co‐op 1.5") == ["oil", "oil", "not", "petroleum", "wheat's", "co-op", "1.5"]


def test_train_only_idf_actual_tf_and_test_oov():
    space = TfidfSpace.fit([["oil", "oil", "gas"], ["gas"], []])
    assert space.n_train == 3
    oil, gas = space.positions["oil"], space.positions["gas"]
    assert space.df[oil] == 1 and space.cf[oil] == 2
    matrix = space.transform([["oil", "oil", "new-test-only"]]).toarray()[0]
    assert matrix[oil] == pytest.approx(2 * math.log10(3))
    assert matrix[gas] == 0 and "new-test-only" not in space.positions
    assert space.df[oil] == 1


def test_weighted_embedding_matches_hand_calculation_and_reports_oov_weight():
    space = TfidfSpace.fit([["oil"], ["gas"], ["missing"]])
    aligned = space.word_matrix(["oil", "gas"], np.array([[1., 0.], [0., 2.]]))
    vectors, coverage = weighted_documents(space.transform([["oil", "oil", "gas", "missing"], []]), aligned)
    assert vectors[0] == pytest.approx([2/3, 2/3])
    assert coverage == pytest.approx([3/4, 0])
    assert not np.any(vectors[1])


@pytest.fixture
def semantic_engine():
    # The best semantic document has zero exact token overlap with the query.
    space = TfidfSpace.fit([["oil"], ["petroleum"], ["loss"], ["other"]])
    docs = [{"id": "20", "title": "Petroleum", "body": "", "topics": ["crude"]},
            {"id": "3", "title": "Loss", "body": "", "topics": ["earn"]},
            {"id": "4", "title": "", "body": "", "topics": []}]
    matrix = space.transform([["petroleum"], ["loss"], []])
    models = {name: (["oil", "petroleum", "loss"], np.array([[1., 0.], [1., 0.], [-1., 0.]]))
              for name in ("cbow", "skipgram")}
    vectors = {name: weighted_documents(matrix, space.word_matrix(*model))[0] for name, model in models.items()}
    return RetrievalEngine(space, docs, matrix, models, vectors)


def test_semantic_retrieval_has_no_lexical_gate_and_keeps_negative_cosine(semantic_engine):
    ranked, status = semantic_engine.rank("oil", "cbow")
    assert [r["id"] for r in ranked] == ["20", "3"]
    assert [r["score"] for r in ranked] == pytest.approx([1., -1.])
    assert status["valid_documents"] == 2
    assert "4" not in [r["id"] for r in ranked]


def test_unweighted_retrieval_counts_repeated_terms_and_handles_missing_vectors(semantic_engine):
    from dataclasses import replace
    from ir_hw4.embedding_weighting import unweighted_scores

    space = replace(semantic_engine.space)
    space.idf.fill(1.)
    documents = space.transform([["petroleum"], ["loss"], []])
    tokens = [["oil", "oil", "loss", "missing"], ["missing"]]
    before, oov = unweighted_scores(semantic_engine, "cbow", documents, space.transform(tokens), tokens, [{"3"}] * 2)
    assert before["valid_query"] and before["valid_documents"] == 2
    assert before["mrr"] == 0.5 and before["p10"] == 0.1 and before["missing_terms"] == ["missing"]
    assert not oov["valid_query"] and oov["ndcg10"] == 0


@pytest.mark.parametrize("query", ["", "the and", "zzzzunknown"])
@pytest.mark.parametrize("method", ["tfidf", "cbow", "skipgram"])
def test_empty_or_oov_query_has_no_fabricated_ranking(semantic_engine, query, method):
    ranked, status = semantic_engine.rank(query, method)
    assert ranked == [] and not status["valid_query"]


def test_stable_tie_break_uses_numeric_ids():
    assert ordered_indices(np.array([0.5, 0.5 + 1e-14, -0.1]), ["20", "3", "10"]).tolist() == [1, 0, 2]


def test_neighbors_exclude_self_and_handle_missing_and_zero_vectors():
    words = ["oil", "petroleum", "loss", "empty"]
    vectors = np.array([[1., 0.], [1., 0.], [-1., 0.], [0., 0.]])
    assert [r["word"] for r in neighbors(words, vectors, "oil")] == ["petroleum", "loss"]
    assert neighbors(words, vectors, "missing") == []
    assert neighbors(words, vectors, "empty") == []


def test_evaluation_known_hand_values_full_rank_rr_and_fixed_precision_denominator():
    result = metrics(["x", "r", "z", "s"], {"r", "s"}, 10)
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert result["p10"] == 0.2
    assert result["mrr"] == 0.5
    assert result["ndcg10"] == pytest.approx(expected_dcg / (1 + 1 / math.log2(3)))
    assert metrics(["x"] * 0 + [str(i) for i in range(10)] + ["r"], {"r"})["mrr"] == 1 / 11
    assert metrics([], {"r"})["p10"] == 0
    assert metrics([], set())["p10"] is None
    with pytest.raises(ValueError):
        metrics(["r", "r"], {"r"})
