import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import transformers  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="transformers not installed")
        for it in items:
            if "lm" in it.keywords:
                it.add_marker(skip)
