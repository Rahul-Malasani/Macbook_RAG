from clirag.loader import load_documents


def test_loads_corpus():
    docs = load_documents()
    # The harvested man-page corpus is committed, so this runs in CI too.
    assert len(docs) > 50, "expected the harvested man-page corpus to be present"


def test_document_shape():
    d = load_documents()[0]
    assert d.text.strip(), "document text should be non-empty"
    assert d.metadata["tool"], "each document should carry its tool name"
    assert d.metadata["source"].endswith(".txt")


def test_known_tools_present():
    tools = {d.metadata["tool"] for d in load_documents()}
    # Sanity: a few tools we know we harvested, including confusable-cluster members.
    assert {"grep", "scp", "git", "find"} <= tools
