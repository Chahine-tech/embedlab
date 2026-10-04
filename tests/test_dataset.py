"""Dataset loading refuses quietly-wrong input and reports the offending line."""

from pathlib import Path

import pytest

from embedlab.artifacts.dataset import DatasetError, load_dataset
from embedlab.domain.ids import DocId, QueryId

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def write(directory: Path, corpus: str = "", queries: str = "", qrels: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.jsonl").write_text(corpus)
    (directory / "queries.jsonl").write_text(queries)
    (directory / "qrels.jsonl").write_text(qrels)
    return directory


VALID_CORPUS = '{"doc_id": "d1", "text": "alpha"}\n{"doc_id": "d2", "text": "beta"}\n'
VALID_QUERIES = '{"query_id": "q1", "text": "alpha?"}\n'
VALID_QRELS = '{"query_id": "q1", "doc_id": "d1", "relevance": 2}\n'


def test_loads_the_mini_fixture():
    dataset = load_dataset(FIXTURE)
    assert len(dataset.corpus) == 20
    assert len(dataset.queries) == 7
    assert len(dataset.judged_queries) == 7
    assert dataset.qrels[QueryId("q-worker-crash")][DocId("retry-policy")] == 1


def test_corpus_fingerprint_ignores_labels(tmp_path):
    """Re-labelling must not invalidate cached embeddings."""
    first = load_dataset(write(tmp_path / "a", VALID_CORPUS, VALID_QUERIES, VALID_QRELS))
    relabelled = '{"query_id": "q1", "doc_id": "d2", "relevance": 1}\n'
    second = load_dataset(write(tmp_path / "b", VALID_CORPUS, VALID_QUERIES, relabelled))

    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.labels_fingerprint != second.labels_fingerprint


def test_corpus_fingerprint_changes_with_content(tmp_path):
    first = load_dataset(write(tmp_path / "a", VALID_CORPUS, VALID_QUERIES, VALID_QRELS))
    altered = '{"doc_id": "d1", "text": "ALPHA changed"}\n{"doc_id": "d2", "text": "beta"}\n'
    second = load_dataset(write(tmp_path / "b", altered, VALID_QUERIES, VALID_QRELS))
    assert first.corpus_fingerprint != second.corpus_fingerprint


def test_queries_without_positive_labels_are_not_judged(tmp_path):
    queries = '{"query_id": "q1", "text": "a?"}\n{"query_id": "q2", "text": "b?"}\n'
    qrels = VALID_QRELS + '{"query_id": "q2", "doc_id": "d2", "relevance": 0}\n'
    dataset = load_dataset(write(tmp_path, VALID_CORPUS, queries, qrels))
    assert dataset.judged_queries == (QueryId("q1"),)


@pytest.fixture
def directory(tmp_path):
    return tmp_path


def test_missing_file_is_reported(directory):
    with pytest.raises(DatasetError, match="missing dataset file"):
        load_dataset(directory)


def test_malformed_json_names_the_line(directory):
    write(directory, VALID_CORPUS + "{not json\n", VALID_QUERIES, VALID_QRELS)
    with pytest.raises(DatasetError, match=r"corpus\.jsonl:3"):
        load_dataset(directory)


def test_blank_lines_are_tolerated(directory):
    write(directory, "\n" + VALID_CORPUS + "\n", VALID_QUERIES, VALID_QRELS)
    assert len(load_dataset(directory).corpus) == 2


def test_duplicate_doc_id_is_refused(directory):
    write(
        directory, VALID_CORPUS + '{"doc_id": "d1", "text": "again"}\n', VALID_QUERIES, VALID_QRELS
    )
    with pytest.raises(DatasetError, match="duplicate doc_id"):
        load_dataset(directory)


def test_label_for_unknown_query_is_refused(directory):
    write(directory, VALID_CORPUS, VALID_QUERIES, '{"query_id": "ghost", "doc_id": "d1"}\n')
    with pytest.raises(DatasetError, match="unknown query_id"):
        load_dataset(directory)


def test_label_for_unknown_doc_is_refused(directory):
    """The silent-corruption case: a qrel pointing at a document that is gone."""
    write(directory, VALID_CORPUS, VALID_QUERIES, '{"query_id": "q1", "doc_id": "ghost"}\n')
    with pytest.raises(DatasetError, match="unknown doc_id"):
        load_dataset(directory)


def test_non_integer_relevance_is_refused(directory):
    write(
        directory,
        VALID_CORPUS,
        VALID_QUERIES,
        '{"query_id": "q1", "doc_id": "d1", "relevance": 1.5}\n',
    )
    with pytest.raises(DatasetError, match="must be an integer"):
        load_dataset(directory)


def test_empty_text_is_refused(directory):
    write(directory, '{"doc_id": "d1", "text": ""}\n', VALID_QUERIES, VALID_QRELS)
    with pytest.raises(DatasetError, match="non-empty string"):
        load_dataset(directory)


def test_relevance_defaults_to_one(directory):
    write(directory, VALID_CORPUS, VALID_QUERIES, '{"query_id": "q1", "doc_id": "d1"}\n')
    assert load_dataset(directory).qrels[QueryId("q1")][DocId("d1")] == 1


def test_empty_corpus_is_refused(directory):
    write(directory, "", VALID_QUERIES, "")
    with pytest.raises(DatasetError, match="corpus is empty"):
        load_dataset(directory)
