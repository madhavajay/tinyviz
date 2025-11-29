import pytest
from tinygrad import Tensor
from tinyviz import (
  _flatten,
  _norm,
  _mix,
  _to_matrix,
  _matrix_dims,
  _shape,
  _op_name,
  name_tensor,
)
from colour import Color


class TestFlatten:
  def test_none(self):
    assert _flatten(None) == []

  def test_single_int(self):
    assert _flatten(5) == [5.0]

  def test_single_float(self):
    assert _flatten(3.14) == [3.14]

  def test_bool_converts_to_int(self):
    assert _flatten(True) == [1]
    assert _flatten(False) == [0]

  def test_flat_list(self):
    assert _flatten([1, 2, 3]) == [1.0, 2.0, 3.0]

  def test_nested_list(self):
    assert _flatten([[1, 2], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]

  def test_deeply_nested(self):
    assert _flatten([[[1, 2]], [[3, 4]]]) == [1.0, 2.0, 3.0, 4.0]


class TestNorm:
  def test_zero_value(self):
    assert _norm(0, -10, 10) == 0

  def test_positive_value(self):
    result = _norm(5, -10, 10)
    assert 0 <= result <= 1
    assert result == 0.5

  def test_negative_value(self):
    result = _norm(-5, -10, 10)
    assert 0 <= result <= 1
    assert result == 0.5

  def test_none_bounds(self):
    assert _norm(5, None, None) == 0
    assert _norm(5, None, 10) == 0
    assert _norm(5, -10, None) == 0

  def test_max_value(self):
    assert _norm(10, -10, 10) == 1.0

  def test_exceeds_max(self):
    assert _norm(20, -10, 10) == 1.0

  def test_zero_bounds(self):
    assert _norm(0, 0, 0) == 0


class TestMix:
  def test_zero_interpolation(self):
    c1 = Color("red")
    c2 = Color("blue")
    result = _mix(c1, c2, 0.0)
    assert result.red == c1.red
    assert result.green == c1.green
    assert result.blue == c1.blue

  def test_full_interpolation(self):
    c1 = Color("red")
    c2 = Color("blue")
    result = _mix(c1, c2, 1.0)
    assert result.red == c2.red
    assert result.green == c2.green
    assert result.blue == c2.blue

  def test_half_interpolation(self):
    c1 = Color("black")
    c2 = Color("white")
    result = _mix(c1, c2, 0.5)
    assert 0.4 < result.red < 0.6
    assert 0.4 < result.green < 0.6
    assert 0.4 < result.blue < 0.6

  def test_clamps_above_one(self):
    c1 = Color("red")
    c2 = Color("blue")
    result = _mix(c1, c2, 2.0)
    assert result.red == c2.red

  def test_clamps_below_zero(self):
    c1 = Color("red")
    c2 = Color("blue")
    result = _mix(c1, c2, -1.0)
    assert result.red == c1.red


class TestToMatrix:
  def test_none(self):
    assert _to_matrix(None) == [[None]]

  def test_scalar(self):
    assert _to_matrix(5) == [[5]]
    assert _to_matrix(3.14) == [[3.14]]
    assert _to_matrix(True) == [[True]]

  def test_flat_list(self):
    assert _to_matrix([1, 2, 3]) == [[1, 2, 3]]

  def test_matrix(self):
    assert _to_matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


class TestMatrixDims:
  def test_scalar(self):
    mat, rows, cols = _matrix_dims(5)
    assert mat == [[5]]
    assert rows == 1
    assert cols == 1

  def test_vector(self):
    mat, rows, cols = _matrix_dims([1, 2, 3])
    assert mat == [[1, 2, 3]]
    assert rows == 1
    assert cols == 3

  def test_matrix(self):
    mat, rows, cols = _matrix_dims([[1, 2], [3, 4], [5, 6]])
    assert mat == [[1, 2], [3, 4], [5, 6]]
    assert rows == 3
    assert cols == 2

  def test_none(self):
    mat, rows, cols = _matrix_dims(None)
    assert mat == [[None]]
    assert rows == 1
    assert cols == 1


class TestShape:
  def test_tensor_shape(self):
    t = Tensor([[1, 2], [3, 4]])
    assert _shape(t) == (2, 2)

  def test_vector_tensor(self):
    t = Tensor([1, 2, 3])
    assert _shape(t) == (3,)

  def test_scalar_tensor(self):
    t = Tensor(5)
    assert _shape(t) == tuple()

  def test_non_tensor(self):
    assert _shape(5) is None
    assert _shape([1, 2, 3]) is None


class TestOpName:
  def test_has_name_attribute(self):
    class MockOp:
      name = "TEST_OP"

    assert _op_name(MockOp()) == "TEST_OP"

  def test_no_name_attribute(self):
    class MockOp:
      def __str__(self):
        return "MOCK_OP"

    assert _op_name(MockOp()) == "MOCK_OP"


class TestNameTensor:
  def test_sets_name(self):
    t = Tensor([1, 2, 3])
    name_tensor(t, "my_tensor")
    assert getattr(t, "_tg_name") == "my_tensor"

  def test_returns_tensor(self):
    t = Tensor([1, 2, 3])
    result = name_tensor(t, "my_tensor")
    assert result is t

  def test_non_tensor(self):
    result = name_tensor(5, "not_a_tensor")
    assert result == 5
