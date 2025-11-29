import pytest
from tinyviz import reset_graph


@pytest.fixture(autouse=True)
def reset_before_each():
  reset_graph()
  yield
  reset_graph()
