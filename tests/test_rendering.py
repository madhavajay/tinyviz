import pytest
from tinygrad import Tensor
from tinyviz import (
  graph,
  reset_graph,
  render_graph,
  _color_for,
  _palette,
  dark,
  light,
)
from colour import Color


class TestPalette:
  def test_dark_palette(self):
    dark()
    pal = _palette()
    assert pal["bg"] == Color("black")
    assert pal["text"] == Color("white")

  def test_light_palette(self):
    light()
    pal = _palette()
    assert pal["bg"] == Color("white")
    assert pal["text"] == Color("black")


class TestColorFor:
  def setup_method(self):
    dark()

  def test_zero_value(self):
    color, opacity = _color_for(0, -10, 10)
    assert opacity == 0.45

  def test_positive_value(self):
    color, opacity = _color_for(10, -10, 10)
    assert 0.45 < opacity <= 1.0

  def test_negative_value(self):
    color, opacity = _color_for(-10, -10, 10)
    assert 0.45 < opacity <= 1.0

  def test_color_is_color_object(self):
    color, opacity = _color_for(5, -10, 10)
    assert isinstance(color, Color)

  def test_opacity_in_range(self):
    color, opacity = _color_for(7, -10, 10)
    assert 0.45 <= opacity <= 1.0


class TestRenderGraph:
  def setup_method(self):
    reset_graph()

  def test_empty_graph(self):
    diagram = render_graph([])
    assert diagram is None

  def test_single_operation(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    diagram = render_graph(graph)
    assert diagram is not None

  def test_multiple_operations(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a + b
    d = c * a
    diagram = render_graph(graph)
    assert diagram is not None

  def test_with_title(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    diagram = render_graph(graph, title="Test Graph")
    assert diagram is not None

  def test_missing_values(self):
    nodes = [
      {
        "id": 1,
        "op": "ADD",
        "inputs": [(2,), (2,)],
        "input_names": ["a", "b"],
        "result_name": "c",
      }
    ]
    diagram = render_graph(nodes, title="Incomplete Node")
    assert diagram is not None


class TestGraphAPI:
  def test_dark_returns_graph(self):
    result = dark()
    assert result is graph

  def test_light_returns_graph(self):
    result = light()
    assert result is graph


class TestComplexOperations:
  def setup_method(self):
    reset_graph()

  def test_chained_operations(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    d = c * 2
    assert len(graph) >= 2

  def test_matrix_multiplication(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a * b
    assert len(graph) == 1
    assert graph[0]["inputs"][0] == (2, 2)
    assert graph[0]["inputs"][1] == (2, 2)

  def test_broadcasting(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([10, 20])
    c = a + b
    assert len(graph) == 1
