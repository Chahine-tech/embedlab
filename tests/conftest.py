import pytest

from embedlab.artifacts.dataset import load_dataset


def pytest_configure(config):
    config.addinivalue_line("markers", "lexical: needs the 'lexical' extra")
    config.addinivalue_line("markers", "measures: needs the 'measures' extra")


@pytest.fixture
def mini(request):
    return load_dataset(request.path.parent / "fixtures" / "mini")
