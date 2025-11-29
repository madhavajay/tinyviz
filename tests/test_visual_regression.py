import pytest
from pathlib import Path
from tinygrad import Tensor
from tinyviz import graph, reset_graph, save_image


class TestVisualRegression:
    def test_notebook_example_generates_graph(self):
        """
        Visual regression test based on test.ipynb.
        Generates graph and saves to tests/fixtures/expected_graph.png
        If the graph looks wrong, this indicates broken code.
        """
        reset_graph()

        # From test.ipynb
        a1 = Tensor([[1, 2], [1, 3]])
        b1 = Tensor([-10, 255])

        def add_stuff(a, b):
            c = a * b
            d = a * c
            return d

        z = a1 * a1
        result = add_stuff(a1, b1)

        # Verify operations were recorded
        assert len(graph) >= 3

        # Save graph for visual inspection
        fixtures_dir = Path(__file__).parent / "fixtures"
        fixtures_dir.mkdir(exist_ok=True)
        output_path = fixtures_dir / "test_graph_output.png"

        save_image(str(output_path), title="Test Graph", scale=2.0)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be reasonably sized

    def test_simple_operations_graph(self):
        """Generate a simple graph for visual verification"""
        reset_graph()

        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a + b
        d = c * a

        assert len(graph) == 2

        fixtures_dir = Path(__file__).parent / "fixtures"
        fixtures_dir.mkdir(exist_ok=True)
        output_path = fixtures_dir / "simple_operations.png"

        save_image(str(output_path), title="Simple Operations", scale=2.0)
        assert output_path.exists()
