import pytest
from tinygrad import Tensor
from tinyviz import (
  graph,
  reset_graph,
  name_tensor,
  infer_names_from_frame,
  _ensure_name,
)


class TestNameTensor:
  def test_basic_naming(self):
    t = Tensor([1, 2, 3])
    name_tensor(t, "foo")
    assert t._tg_name == "foo"

  def test_name_appears_in_graph(self):
    reset_graph()
    a = Tensor([1, 2])
    name_tensor(a, "input_a")
    b = Tensor([3, 4])
    name_tensor(b, "input_b")
    c = a + b
    input_names = graph[0]["input_names"]
    assert input_names[0] == "input_a" or input_names[0] == "a"
    assert input_names[1] == "input_b" or input_names[1] == "b"


class TestInferNamesFromFrame:
  def test_infers_local_names(self):
    my_tensor = Tensor([1, 2, 3])
    another_tensor = Tensor([4, 5, 6])
    infer_names_from_frame(my_tensor, another_tensor, depth=1)
    assert my_tensor._tg_name == "my_tensor"
    assert another_tensor._tg_name == "another_tensor"

  def test_skips_named_tensors(self):
    t = Tensor([1, 2, 3])
    name_tensor(t, "custom_name")
    infer_names_from_frame(t, depth=1)
    assert t._tg_name == "custom_name"

  def test_handles_non_tensors(self):
    infer_names_from_frame(5, "hello", [1, 2, 3], depth=1)


class TestResultNaming:
  def setup_method(self):
    reset_graph()

  def test_composite_name_from_inputs(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    assert graph[0]["result_name"] == "a_ADD_b"

  def test_assignment_name_detection(self):
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    result = a + b
    assert graph[0]["result_name"] is not None

  def test_fallback_to_id(self):
    reset_graph()
    graph.stop()
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    graph.start()
    result = (a + b) * (a - b)
    graph.stop()
    a2 = Tensor([5, 6])
    b2 = Tensor([7, 8])
    graph.start()
    c = a2 + b2
    assert graph[0]["result_name"] in ["c", "a2_ADD_b2"]


class TestCaptureWithNaming:
  def setup_method(self):
    reset_graph()

  def test_parameter_names_captured(self):
    @graph.capture
    def compute(input_x, input_y, scale):
      temp = input_x + input_y
      return temp * scale

    a = Tensor([1, 2])
    b = Tensor([3, 4])
    s = Tensor([10])
    result = compute(a, b, s)
    assert graph[0]["input_names"] == ["input_x", "input_y"]
    assert graph[1]["input_names"][0] == "temp"

  def test_multiple_calls_preserve_names(self):
    @graph.capture
    def add(x, y):
      return x + y

    reset_graph()
    a = Tensor([1])
    b = Tensor([2])
    c = add(a, b)
    assert graph[0]["input_names"] == ["x", "y"]

    reset_graph()
    d = Tensor([3])
    e = Tensor([4])
    f = add(d, e)
    assert graph[0]["input_names"] == ["x", "y"]
