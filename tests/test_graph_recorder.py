import pytest
from tinygrad import Tensor
from tinyviz import graph, reset_graph, GraphRecorder, record_graph


class TestGraphRecorder:
  def setup_method(self):
    reset_graph()

  def test_start_clears_and_activates(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert len(graph) == 1
    graph.start()
    assert len(graph) == 0

  def test_stop_deactivates_recording(self):
    graph.start()
    graph.stop()
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert len(graph) == 0

  def test_capture_decorator(self):
    @graph.capture
    def add_tensors(x, y):
      return x + y

    reset_graph()
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    result = add_tensors(a, b)
    assert len(graph) == 1
    assert graph[0]["input_names"] == ["x", "y"]

  def test_dark_mode(self):
    from tinyviz import _theme
    graph.light()
    assert _theme["dark"] is False
    graph.dark()
    assert _theme["dark"] is True

  def test_light_mode(self):
    from tinyviz import _theme
    graph.dark()
    assert _theme["dark"] is True
    graph.light()
    assert _theme["dark"] is False


class TestRecordGraph:
  def test_context_manager(self):
    reset_graph()
    with record_graph() as g:
      a = Tensor([1, 2])
      b = Tensor([3, 4])
      c = a + b
      assert len(g) == 1

    d = Tensor([5, 6])
    e = Tensor([7, 8])
    f = d + e
    assert len(g) == 1

  def test_context_manager_resets_on_entry(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert len(graph) == 1

    with record_graph() as g:
      assert len(g) == 0

  def test_graph_as_context_manager(self):
    reset_graph()
    with graph:
      a = Tensor([1, 2])
      b = Tensor([3, 4])
      c = a + b
      assert len(graph) == 1

    d = Tensor([5, 6])
    e = Tensor([7, 8])
    f = d + e
    assert len(graph) == 1


class TestBasicRecording:
  def setup_method(self):
    reset_graph()

  def test_records_addition(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert len(graph) == 1
    assert graph[0]["op"] == "ADD"

  def test_records_multiplication(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a * b
    assert len(graph) == 1
    assert graph[0]["op"] == "MUL"

  def test_records_multiple_ops(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    d = c * a
    assert len(graph) == 2
    assert graph[0]["op"] == "ADD"
    assert graph[1]["op"] == "MUL"

  def test_captures_shapes(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a + b
    assert graph[0]["inputs"][0] == (2, 2)
    assert graph[0]["inputs"][1] == (2, 2)
    assert graph[0]["result_shape"] == (2, 2)

  def test_captures_values(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert graph[0]["inputs_value"][0] == [1, 2]
    assert graph[0]["inputs_value"][1] == [3, 4]
    assert graph[0]["result_value"] == [4, 6]

  def test_increments_ids(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    d = c * a
    assert graph[0]["id"] == 1
    assert graph[1]["id"] == 2

  def test_reset_graph_clears_ids(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert graph[0]["id"] == 1

    reset_graph()
    d = Tensor([5, 6])
    e = Tensor([7, 8])
    f = d + e
    assert graph[0]["id"] == 1
