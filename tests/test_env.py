def test_imports():
    import importlib

    assert importlib.util.find_spec("transformer") is not None
    assert importlib.util.find_spec("tokenizer") is not None


def test_python_version():
    import sys

    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 10
