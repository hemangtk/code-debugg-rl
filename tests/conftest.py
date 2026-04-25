"""Pytest config — tell pytest not to collect Pydantic models named Test*."""
collect_ignore_glob = []


def pytest_collection_modifyitems(config, items):
    return


def pytest_collectstart(collector):
    return


# The actual fix: tell pytest those classes aren't tests.
import server.models as _models
for _name in ("TestCase", "TestResult"):
    cls = getattr(_models, _name, None)
    if cls is not None:
        cls.__test__ = False
