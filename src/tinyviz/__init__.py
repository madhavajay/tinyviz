from tinygrad import Tensor
from chalk import hcat, vcat, text, rectangle, vstrut, set_svg_height
from colour import Color
from IPython.display import display, SVG
import io
import os
import inspect
import importlib
from inspect import currentframe

_id = {"next": 0}
_record_active = {"on": True}
_theme = {"dark": True}


class GraphRecorder(list):
    def start(self):
        reset_graph()
        _record_active["on"] = True
        return self

    def stop(self):
        _record_active["on"] = False
        return self

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def _repr_png_(self):
        try:
            png_bytes = _draw_pil(self, title="graph", path=None, scale=2.0)
            if png_bytes is None:
                return None
            # Return in IPython.display.Image format for better rendering
            from IPython.display import Image

            return Image(data=png_bytes)._repr_png_()
        except Exception:
            return None

    def _repr_svg_(self):
        # Disabled: PNG rendering looks better and is more consistent
        # Jupyter will fall back to _repr_png_() which matches show_graph()
        return None

    def capture(self, fn):
        """Decorator: wraps fn so recording turns on when called and off when it returns."""

        def wrapped(*args, **kwargs):
            # tag tensor args with parameter names
            try:
                sig = inspect.signature(fn)
                ba = sig.bind_partial(*args, **kwargs)
                ba.apply_defaults()
                for name, val in ba.arguments.items():
                    if isinstance(val, Tensor):
                        setattr(val, "_tg_name", name)
            except Exception:
                pass
            self.start()
            try:
                return fn(*args, **kwargs)
            finally:
                self.stop()

        return wrapped

    def dark(self):
        _theme["dark"] = True
        return self

    def light(self):
        _theme["dark"] = False
        return self


graph = GraphRecorder()


def _next_id():
    _id["next"] += 1
    return _id["next"]


def _op_name(op):
    return op.name if hasattr(op, "name") else str(op)


def _ensure_name(tensor, skip_frames=1, debug=False):
    """Find the variable name for a tensor by inspecting the calling frame."""
    if not isinstance(tensor, Tensor):
        return None
    try:
        frame = currentframe()
        # Skip our own frame and any requested frames
        for _ in range(skip_frames + 1):
            if frame:
                frame = frame.f_back

        # Search frames from most recent to oldest
        frame_idx = 0
        while frame:
            mod = frame.f_globals.get("__name__", "")
            fname = frame.f_code.co_filename
            if debug:
                print(
                    f"    Frame {frame_idx}: {mod}, func={frame.f_code.co_name}, file={fname}"
                )
            # Skip only tinygraph.py or tinygrad package frames
            if (
                fname.endswith("tinygraph.py")
                or "/tinygrad/" in fname
                or "\\tinygrad\\" in fname
            ):
                if debug:
                    print("      Skipping tinygraph/tinygrad frame")
                frame = frame.f_back
                frame_idx += 1
                continue

            # Look for the tensor in this frame's locals
            for k, v in frame.f_locals.items():
                if k == "self" or k.startswith("_"):
                    continue
                if v is tensor:
                    if debug:
                        print(f"      Found tensor as '{k}'")
                    return k

            # Move to parent frame
            frame = frame.f_back
            frame_idx += 1
    except Exception as e:
        if debug:
            print(f"    Exception: {e}")
        pass

    # Fall back to previously set name
    prev_name = getattr(tensor, "_tg_name", None)
    if debug and prev_name:
        print(f"    Using previous name: {prev_name}")
    return prev_name


def _stack_context(max_entries: int = 2, debug=False):
    ctx = []
    try:
        for idx, fi in enumerate(inspect.stack()):
            fname = fi.function
            path = fi.filename or ""
            if debug:
                print(f"    Stack[{idx}]: {fname} in {path}")
                if fi.code_context:
                    print(f"      Code: {fi.code_context[0].strip()}")
            # Skip only actual tinygraph.py or tinygrad package files, not user files with those names in path
            if (
                path.endswith("tinygraph.py")
                or "/tinygrad/" in path
                or "\\tinygrad\\" in path
            ):
                if debug:
                    print("      Skipping (tinygraph/tinygrad module)")
                continue
            args = []
            try:
                arginfo = inspect.getargvalues(fi.frame)
                args = list(arginfo.args or [])
            except Exception:
                pass
            sig = f"{fname}({', '.join(args)})"
            code_line = ""
            assign = None
            if fi.code_context:
                code_line = fi.code_context[0].strip()
                if (
                    "=" in code_line
                    and "==" not in code_line
                    and "!=" not in code_line
                    and "<=" not in code_line
                    and ">=" not in code_line
                ):
                    lhs = code_line.split("=", 1)[0].strip()
                    if lhs and all(ch.isalnum() or ch == "_" for ch in lhs):
                        assign = lhs
                        if debug:
                            print(f"      Found assignment: {assign}")
            ctx.append({"sig": sig, "code": code_line, "assign": assign})
            if len(ctx) >= max_entries:
                break
    except Exception:
        pass
    return ctx


def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return None


orig_binop = Tensor._binop


def _binop_patch(self, op, x, reverse):
    res = orig_binop(self, op, x, reverse)

    # Get stack context first to find assignment target
    debug = False  # Set to True for debugging
    stack_ctx = _stack_context(debug=debug)
    assign_name = stack_ctx[0].get("assign") if stack_ctx else None

    # Get input names from current context (skip 2 frames: this one + operator wrapper)
    if debug:
        print(f"\n_binop_patch: {_op_name(op)}")
        print("  Finding name for self:")
    in_names = [
        _ensure_name(self, skip_frames=2, debug=debug),
        _ensure_name(x, skip_frames=2, debug=debug) if isinstance(x, Tensor) else None,
    ]
    if debug and isinstance(x, Tensor):
        print("  Finding name for x:")

    # If recording is off, just set name and return
    if not _record_active["on"]:
        # Prefer assignment target, then composite name
        if assign_name:
            res_name = assign_name
        elif any(in_names):
            left = in_names[0] or ""
            right = in_names[1] or ""
            res_name = f"{left}_{_op_name(op)}_{right}".strip("_") or None
        else:
            res_name = None
        if res_name:
            setattr(res, "_tg_name", res_name)
        return res

    # Recording is on - build node
    node = {
        "id": _next_id(),
        "op": _op_name(op),
        "inputs": [_shape(self), _shape(x) if isinstance(x, Tensor) else None],
        "input_names": in_names,
        "stack": stack_ctx,
        "assign": assign_name,
        "signature": stack_ctx[0].get("sig") if stack_ctx else None,
        "expression": stack_ctx[0].get("code") if stack_ctx else None,
    }
    try:
        node["result_shape"] = _shape(res)
        node["result_value"] = res.numpy().tolist()
        node["inputs_value"] = [
            self.numpy().tolist(),
            x.numpy().tolist() if isinstance(x, Tensor) else x,
        ]
    except Exception as e:
        node["error"] = str(e)

    # Name the result: prefer assignment target, then composite name
    if assign_name:
        res_name = assign_name
    elif any(in_names):
        left = in_names[0] or ""
        right = in_names[1] or ""
        res_name = f"{left}_{_op_name(op)}_{right}".strip("_")
    else:
        res_name = f"t{node['id']}"

    setattr(res, "_tg_name", res_name)
    node["result_name"] = res_name
    graph.append(node)
    return res


orig_unary = getattr(Tensor, "_unary", None)


def _unary_patch(self, op):
    res = orig_unary(self, op)

    # Get stack context and input name
    stack_ctx = _stack_context()
    assign_name = stack_ctx[0].get("assign") if stack_ctx else None
    input_name = _ensure_name(self, skip_frames=2)

    # If recording is off, just set name and return
    if not _record_active["on"]:
        res_name = assign_name or input_name
        if res_name:
            setattr(res, "_tg_name", res_name)
        return res

    # Recording is on - build node
    node = {
        "id": _next_id(),
        "op": f"UNARY_{_op_name(op)}",
        "inputs": [_shape(self)],
        "input_names": [input_name],
        "stack": stack_ctx,
        "assign": assign_name,
        "signature": stack_ctx[0].get("sig") if stack_ctx else None,
        "expression": stack_ctx[0].get("code") if stack_ctx else None,
    }
    try:
        node["result_shape"] = _shape(res)
        node["result_value"] = res.numpy().tolist()
        node["inputs_value"] = [self.numpy().tolist()]
    except Exception as e:
        node["error"] = str(e)

    # Name the result: prefer assignment target, then input name
    res_name = assign_name or input_name or f"t{node['id']}"
    setattr(res, "_tg_name", res_name)
    node["result_name"] = res_name
    graph.append(node)
    return res


def enable():
    """Monkey-patch Tensor ops to record a graph."""
    Tensor._binop = _binop_patch
    if orig_unary:
        Tensor._unary = _unary_patch


def name_tensor(tensor, name: str):
    """Attach a readable name to a Tensor for graph display."""
    if isinstance(tensor, Tensor):
        setattr(tensor, "_tg_name", name)
    return tensor


def infer_names_from_frame(*tensors, depth: int = 1):
    """Attempt to name tensors based on caller locals at a given frame depth."""
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return
        locals_map = frame.f_locals
        for t in tensors:
            if not isinstance(t, Tensor) or getattr(t, "_tg_name", None):
                continue
            for k, v in locals_map.items():
                if v is t:
                    setattr(t, "_tg_name", k)
                    break
    except Exception:
        pass


def reset_graph():
    """Clear recorded graph and id counter."""
    graph.clear()
    _id["next"] = 0
    _record_active["on"] = True


class record_graph:
    """Context manager to record ops into graph; resets on entry."""

    def __enter__(self):
        graph.start()
        return graph

    def __exit__(self, exc_type, exc, tb):
        graph.stop()
        return False


def start_graph():
    """Start recording (resets existing graph)."""
    return graph.start()


def stop_graph():
    """Stop recording."""
    return graph.stop()


def dark():
    """Set dark theme."""
    _theme["dark"] = True
    return graph


def light():
    """Set light theme."""
    _theme["dark"] = False
    return graph


def save_image(path: str = "graph.png", title: str = "graph", scale: float = 2.0):
    """Save current graph to an image (png/svg based on extension)."""
    if path.lower().endswith(".png"):
        return _draw_pil(graph, title=title, path=path, scale=scale)
    else:
        return save_graph(graph, title=title, path=path)


def _flatten(vals):
    if vals is None:
        return []
    if isinstance(vals, (int, float, bool)):
        return [int(vals) if isinstance(vals, bool) else float(vals)]
    out = []
    for v in vals:
        out.extend(_flatten(v))
    return out


def _norm(v, lo, hi):
    if hi is None or lo is None:
        return 0
    if v == 0 or (hi == 0 and lo == 0):
        return 0
    if v > 0:
        top = hi if hi != 0 else abs(lo) if lo != 0 else 1
        return min(1.0, max(0.0, v / top))
    if v < 0:
        bot = abs(lo) if lo != 0 else hi if hi != 0 else 1
        return min(1.0, max(0.0, abs(v) / bot))
    return 0


def _mix(c1: Color, c2: Color, t: float) -> Color:
    t = min(1.0, max(0.0, t))
    return Color(
        rgb=(
            c1.red + (c2.red - c1.red) * t,
            c1.green + (c2.green - c1.green) * t,
            c1.blue + (c2.blue - c1.blue) * t,
        )
    )


def _palette():
    if _theme["dark"]:
        return {
            "bg": Color("black"),
            "text": Color("white"),
            "label": Color("#bbbbbb"),
            "neutral": Color("#2b2b2b"),
            "cold": Color("#38bdf8"),
            "hot": Color("#f59e0b"),
        }
    else:
        return {
            "bg": Color("white"),
            "text": Color("black"),
            "label": Color("#444444"),
            "neutral": Color("#e6e6e6"),
            "cold": Color("#228be6"),
            "hot": Color("#f59e0b"),
        }


def _color_for(v, lo, hi):
    pal = _palette()
    # Diverging gradient: blue (neg) -> neutral -> orange (pos).
    neutral = pal["neutral"]
    cold = pal["cold"]
    hot = pal["hot"]
    if v == 0 or (lo == 0 and hi == 0):
        return neutral, 0.45
    span = max(abs(lo or 0), abs(hi or 0), 1e-6)
    t = max(-1.0, min(1.0, v / span))
    amt = abs(t)
    if t >= 0:
        col = _mix(neutral, hot, amt)
    else:
        col = _mix(neutral, cold, amt)
    opacity = 0.45 + 0.55 * amt
    return col, opacity


def _to_matrix(v):
    if v is None:
        return [[None]]
    if isinstance(v, (int, float, bool)):
        return [[v]]
    if not any(isinstance(i, list) for i in v):
        return [v]
    return v


def _box(value, lo, hi):
    c, opacity = _color_for(value if value is not None else 0, lo, hi)
    base = rectangle(1.2, 1.2).fill_color(c).fill_opacity(opacity).line_width(0)
    if value is None:
        return base
    label = f"{int(value)}" if float(value).is_integer() else f"{value:.3g}"
    return base + text(label, 0.55).fill_color(_palette()["text"]).line_width(0)


def _draw_matrix(values, lo, hi):
    mat = _to_matrix(values)
    return vcat(
        [hcat([_box(v, lo, hi) for v in row], 0.2) for row in mat],
        0.2,
    )


def render_graph(nodes, title="graph"):
    flat_vals = []
    for n in nodes:
        flat_vals.extend(_flatten(n.get("result_value")))
        flat_vals.extend(
            _flatten((n.get("inputs_value") or [])[0] if n.get("inputs_value") else [])
        )
        if n.get("inputs_value") and len(n["inputs_value"]) > 1:
            flat_vals.extend(_flatten(n["inputs_value"][1]))
    lo = min(flat_vals) if flat_vals else None
    hi = max(flat_vals) if flat_vals else None

    rows = []
    col_max = {"left": 0, "op": 0, "right": 0, "eq": 0, "out": 0}
    col_height = {"left": 0, "op": 0, "right": 0, "eq": 0, "out": 0}

    for n in nodes:
        # Build blocks
        left_vals = (n.get("inputs_value") or [None])[0]
        right_vals = None
        if n.get("inputs_value") and len(n["inputs_value"]) > 1:
            right_vals = n["inputs_value"][1]

        left_name = (n.get("input_names") or [None])[0]
        right_name = (
            (n.get("input_names") or [None, None])[1]
            if len(n.get("input_names") or []) > 1
            else None
        )

        left_block = None
        if left_vals is not None:
            name_txt = (
                text(left_name or "", 0.6).fill_color(_palette()["label"]).line_width(0)
            )
            name_txt = hcat(
                [rectangle(0.15, 0).fill_opacity(0).line_width(0), name_txt], 0.05
            )
            left_block = name_txt / vstrut(0.25) / _draw_matrix(left_vals, lo, hi)
            env = left_block.get_envelope()
            col_max["left"] = max(col_max["left"], env.width)
            col_height["left"] = max(col_height["left"], env.height)

        right_block = None
        if right_vals is not None:
            name_txt = (
                text(right_name or "", 0.6)
                .fill_color(_palette()["label"])
                .line_width(0)
            )
            name_txt = hcat(
                [rectangle(0.15, 0).fill_opacity(0).line_width(0), name_txt], 0.05
            )
            right_block = name_txt / vstrut(0.25) / _draw_matrix(right_vals, lo, hi)
            env = right_block.get_envelope()
            col_max["right"] = max(col_max["right"], env.width)
            col_height["right"] = max(col_height["right"], env.height)

        out_label = n.get("assign") or n.get("result_name") or ""
        out_name = text(out_label, 0.6).fill_color(_palette()["label"]).line_width(0)
        out_name = hcat(
            [rectangle(0.15, 0).fill_opacity(0).line_width(0), out_name], 0.05
        )
        out_block = (
            out_name / vstrut(0.25) / _draw_matrix(n.get("result_value"), lo, hi)
        )
        env = out_block.get_envelope()
        col_max["out"] = max(col_max["out"], env.width)
        col_height["out"] = max(col_height["out"], env.height)

        op_block = text(n["op"], 0.7).fill_color(_palette()["text"]).line_width(0)
        equals_block = text("=", 0.7).fill_color(_palette()["text"]).line_width(0)
        env = op_block.get_envelope()
        col_max["op"] = max(col_max["op"], env.width)
        col_height["op"] = max(col_height["op"], env.height)
        env = equals_block.get_envelope()
        col_max["eq"] = max(col_max["eq"], env.width)
        col_height["eq"] = max(col_height["eq"], env.height)

        label_txt = f"{n['id']}: {n['op']}"
        if n.get("result_name"):
            label_txt += f" → {n['result_name']}"
        header = None

        rows.append(
            dict(
                header=header,
                left=left_block,
                right=right_block,
                out=out_block,
                op=op_block,
                eq=equals_block,
                signature=n.get("signature"),
                expression=n.get("expression"),
            )
        )

    if not rows:
        return None

    uniform_rows = []
    last_sig = None
    for r in rows:
        parts = []
        if r["left"]:
            parts.append(
                r["left"]
                .with_envelope(rectangle(col_max["left"], col_height["left"]))
                .center_xy()
            )
        else:
            parts.append(
                rectangle(col_max["left"], col_height["left"])
                .line_width(0)
                .fill_color(Color("black"))
            )
        parts.append(
            r["op"]
            .with_envelope(rectangle(col_max["op"], col_height["op"]))
            .center_xy()
        )
        if r["right"]:
            parts.append(
                r["right"]
                .with_envelope(rectangle(col_max["right"], col_height["right"]))
                .center_xy()
            )
        else:
            parts.append(
                rectangle(col_max["right"], col_height["right"])
                .line_width(0)
                .fill_color(Color("black"))
            )
        parts.append(
            r["eq"]
            .with_envelope(rectangle(col_max["eq"], col_height["eq"]))
            .center_xy()
        )
        parts.append(
            r["out"]
            .with_envelope(rectangle(col_max["out"], col_height["out"]))
            .center_xy()
        )

        row = hcat(parts, 1.4)
        # Build optional header with signature (when it changes) and expression, plus left label.
        header_block = None
        sig_line = None
        expr_line = None
        if r.get("signature") and r["signature"] != last_sig:
            sig_line = (
                text(r["signature"], 0.42).fill_color(_palette()["label"]).line_width(0)
            )
            last_sig = r["signature"]
        if r.get("expression"):
            expr_line = (
                text(r["expression"], 0.4).fill_color(_palette()["label"]).line_width(0)
            )

        header_parts = [h for h in (sig_line, expr_line) if h]
        if header_parts:
            header_block = vcat(header_parts, 0.05)
            h_env = header_block.get_envelope()
            spacer_w = max(0, row.get_envelope().width - h_env.width)
            header_block = hcat(
                [
                    header_block,
                    rectangle(spacer_w, h_env.height).fill_opacity(0).line_width(0),
                ],
                0,
            )

        blocks_to_add = [b for b in (header_block,) if b]
        if blocks_to_add:
            uniform_rows.append(
                vstrut(0.3) / vcat(blocks_to_add, 0.1) / vstrut(0.4) / row
            )
        else:
            uniform_rows.append(vstrut(0.4) / row)

    max_w = max(b.get_envelope().width for b in uniform_rows)
    uniform_rows = [
        rectangle(max_w, b.get_envelope().height).line_width(0).fill_opacity(0) + b
        for b in uniform_rows
    ]

    diagram = vcat(uniform_rows, 2.4)
    diagram = (
        vstrut(1.6)
        / text(title, 1.1).fill_color(_palette()["text"]).line_width(0)
        / vstrut(1.6)
        / diagram.center_xy()
        / vstrut(1.6)
    )
    diagram = diagram.pad(1.8).center_xy()
    env = diagram.get_envelope()
    set_svg_height(900)
    return (
        rectangle(env.width, env.height).fill_color(_palette()["bg"]).line_width(0)
        + diagram
    )


def _pil_color(c: Color):
    return (
        int(255 * c.red),
        int(255 * c.green),
        int(255 * c.blue),
    )


def _matrix_dims(values):
    mat = _to_matrix(values)
    rows = len(mat)
    cols = max((len(r) for r in mat), default=0)
    return mat, rows, cols


def _draw_pil(nodes, title="graph", path: str | None = "graph.png", scale: float = 1.0):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError("Pillow is required for PNG export. Install pillow.") from e

    def _load_font(size=14):
        # Try a crisp truetype font; fallback to default bitmap.
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    return ImageFont.truetype(c, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    flat_vals = []
    for n in nodes:
        flat_vals.extend(_flatten(n.get("result_value")))
        flat_vals.extend(
            _flatten((n.get("inputs_value") or [])[0] if n.get("inputs_value") else [])
        )
        if n.get("inputs_value") and len(n["inputs_value"]) > 1:
            flat_vals.extend(_flatten(n["inputs_value"][1]))
    lo = min(flat_vals) if flat_vals else None
    hi = max(flat_vals) if flat_vals else None

    # layout constants (scaled for higher DPI)
    cell = 44 * scale
    gap = 10 * scale
    pad = 40 * scale
    row_gap = 32 * scale
    col_gap = 26 * scale
    label_gap = 8 * scale
    pal = _palette()
    bg = tuple(int(255 * c) for c in pal["bg"].get_rgb())
    txt_color = (0, 0, 0) if not _theme["dark"] else (255, 255, 255)
    font = _load_font(size=int(14 * scale))

    draw_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # precompute widths/heights per node (left op right = result)
    rows = []
    max_width = 0
    total_height = pad
    col_max = {"left": 0, "op": 0, "right": 0, "eq": 0, "out": 0}
    col_height = {"left": 0, "op": 0, "right": 0, "eq": 0, "out": 0}

    for n in nodes:
        op_text = n["op"]
        equals_text = "="
        left_name = (n.get("input_names") or [None])[0]
        right_name = (
            (n.get("input_names") or [None, None])[1]
            if len(n.get("input_names") or []) > 1
            else None
        )
        left_val = (n.get("inputs_value") or [None])[0]
        right_val = None
        if n.get("inputs_value") and len(n["inputs_value"]) > 1:
            right_val = n["inputs_value"][1]
        res_name = n.get("result_name")
        res_val = n.get("result_value")

        left_mat, l_rows, l_cols = _matrix_dims(left_val)
        right_mat, r_rows, r_cols = _matrix_dims(right_val)
        res_mat, out_rows, out_cols = _matrix_dims(res_val)

        def mat_size(rows, cols):
            return (
                cols * cell + max(0, cols - 1) * gap,
                rows * cell + max(0, rows - 1) * gap,
            )

        l_w, l_h = mat_size(l_rows, l_cols) if left_val is not None else (0, 0)
        r_w, r_h = mat_size(r_rows, r_cols) if right_val is not None else (0, 0)
        o_w, o_h = mat_size(out_rows, out_cols)

        def text_size(txt):
            if not txt:
                return (0, 0)
            box = draw_dummy.textbbox((0, 0), txt, font=font)
            return (box[2] - box[0], box[3] - box[1])

        llabel_w, llabel_h = text_size(left_name)
        rlabel_w, rlabel_h = text_size(right_name)
        olabel_w, olabel_h = text_size(res_name)
        op_w, op_h = text_size(op_text)
        equals_w, equals_h = text_size(equals_text)

        left_height = (llabel_h + label_gap + l_h) if left_val is not None else 0
        right_height = (rlabel_h + label_gap + r_h) if right_val is not None else 0
        out_height = olabel_h + label_gap + o_h
        op_height = op_h
        eq_height = equals_h
        comp_height = max(left_height, right_height, out_height, op_height, eq_height)

        comp_width = 0
        if left_val is not None:
            comp_width += max(llabel_w, l_w)
            comp_width += col_gap
        comp_width += op_w
        if right_val is not None:
            comp_width += col_gap + max(rlabel_w, r_w)
        comp_width += col_gap + equals_w + col_gap
        comp_width += max(olabel_w, o_w)

        rows.append(
            dict(
                left=dict(
                    name=left_name,
                    mat=left_mat,
                    w=l_w,
                    h=l_h,
                    label_h=llabel_h,
                    show=left_val is not None,
                ),
                right=dict(
                    name=right_name,
                    mat=right_mat,
                    w=r_w,
                    h=r_h,
                    label_h=rlabel_h,
                    show=right_val is not None,
                ),
                out=dict(name=res_name, mat=res_mat, w=o_w, h=o_h, label_h=olabel_h),
                op=dict(text=op_text, w=op_w, h=op_h),
                equals=dict(text=equals_text, w=equals_w, h=equals_h),
                height=comp_height,
                width=comp_width,
            )
        )
        col_max["left"] = max(col_max["left"], max(llabel_w, l_w))
        col_max["right"] = max(col_max["right"], max(rlabel_w, r_w))
        col_max["op"] = max(col_max["op"], op_w)
        col_max["eq"] = max(col_max["eq"], equals_w)
        col_max["out"] = max(col_max["out"], max(olabel_w, o_w))
        col_height["left"] = max(col_height["left"], left_height)
        col_height["right"] = max(col_height["right"], right_height)
        col_height["op"] = max(col_height["op"], op_height)
        col_height["eq"] = max(col_height["eq"], eq_height)
        col_height["out"] = max(col_height["out"], out_height)
        max_width = max(max_width, comp_width)
        total_height += comp_height + row_gap

    if not rows:
        return None

    width = int(max(col_max.values()) * 5 + col_gap * 4 + pad * 2)
    height = int(total_height + pad)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    y = pad
    row_height = max(col_height.values()) if col_height else 0

    for row in rows:
        x = pad
        mid_y = y + row_height // 2

        def draw_matrix(mat, x0, y0, w, h):
            for i, r in enumerate(mat):
                for j, v in enumerate(r):
                    c, _ = _color_for(v if v is not None else 0, lo, hi)
                    color = _pil_color(c)
                    cx = x0 + j * (cell + gap)
                    cy = y0 + i * (cell + gap)
                    draw.rectangle([cx, cy, cx + cell, cy + cell], fill=color)
                    if v is not None:
                        label = f"{int(v)}" if float(v).is_integer() else f"{v:.3g}"
                        tb = draw.textbbox((0, 0), label, font=font)
                        tx = cx + (cell - (tb[2] - tb[0])) / 2
                        ty = cy + (cell - (tb[3] - tb[1])) / 2
                    draw.text((tx, ty), label, fill=txt_color, font=font)

        def draw_label(name, x0, y0):
            if not name:
                return 0, 0
            tb = draw.textbbox((0, 0), name, font=font)
            draw.text((x0, y0), name, fill=(200, 200, 200), font=font)
            return tb[2] - tb[0], tb[3] - tb[1]

        if row["left"]["show"]:
            label_h = row["left"]["label_h"] or 0
            lw, lh = draw_label(row["left"]["name"], x, mid_y - row_height // 2)
            mx = x
            my = mid_y - row_height // 2 + label_h + label_gap
            draw_matrix(row["left"]["mat"], mx, my, row["left"]["w"], row["left"]["h"])
        x += col_max["left"] + col_gap

        # op
        draw.text(
            (x, mid_y - col_height["op"] / 2),
            row["op"]["text"],
            fill=txt_color,
            font=font,
        )
        x += col_max["op"] + col_gap

        if row["right"]["show"]:
            label_h = row["right"]["label_h"] or 0
            rw, rh = draw_label(row["right"]["name"], x, mid_y - row_height // 2)
            mx = x
            my = mid_y - row_height // 2 + label_h + label_gap
            draw_matrix(
                row["right"]["mat"], mx, my, row["right"]["w"], row["right"]["h"]
            )
        x += col_max["right"] + col_gap

        # equals
        draw.text(
            (x, mid_y - col_height["eq"] / 2),
            row["equals"]["text"],
            fill=txt_color,
            font=font,
        )
        x += col_max["eq"] + col_gap

        # output
        label_h = row["out"]["label_h"] or 0
        ow, oh = draw_label(row["out"]["name"], x, mid_y - row_height // 2)
        mx = x
        my = mid_y - row_height // 2 + label_h + label_gap
        draw_matrix(row["out"]["mat"], mx, my, row["out"]["w"], row["out"]["h"])

        y += row["height"] + row_gap

    if path:
        img.save(path)
        return path
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def show_graph(nodes, title="graph"):
    """Render and display the recorded graph; uses PIL for a raster preview."""
    has_pillow = importlib.util.find_spec("PIL.Image") is not None
    if not has_pillow:
        diagram = render_graph(nodes, title=title)
        if diagram is None:
            return None
        svg = SVG(diagram._repr_svg_())
        display(svg)
        return svg
    img_path = _draw_pil(nodes, title=title, path="graph.png", scale=2.0)
    if img_path is None:
        return None
    with open(img_path, "rb") as f:
        from IPython.display import Image as IPyImage

        disp = IPyImage(data=f.read())
        display(disp)
        return disp


def save_graph(nodes, title="graph", path="graph.svg"):
    """Render the graph and save to disk. Supports .svg; tries .png via cairosvg."""
    diagram = render_graph(nodes, title=title)
    if diagram is None:
        return None
    svg_str = diagram._repr_svg_()
    if path.lower().endswith(".svg"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        return path
    if path.lower().endswith(".png"):
        # Try pure-Python PIL fallback for PNG.
        try:
            return _draw_pil(nodes, title=title, path=path, scale=2.0)
        except Exception as e:
            alt = path.rsplit(".", 1)[0] + ".svg"
            with open(alt, "w", encoding="utf-8") as f:
                f.write(svg_str)
            raise RuntimeError(
                f"Failed to write PNG ({e}). SVG saved to {alt}. Install pillow for PNG output."
            ) from e
    # default: write svg
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    return path


enable()


__all__ = [
    "graph",
    "enable",
    "render_graph",
    "show_graph",
    "save_graph",
    "name_tensor",
    "infer_names_from_frame",
    "reset_graph",
    "record_graph",
    "start_graph",
    "stop_graph",
    "dark",
    "light",
    "save_image",
    "capture",
]


def _repr_png_():
    """Default Jupyter repr for the graph as PNG (module-level fallback)."""
    try:
        return _draw_pil(graph, title="graph", path=None, scale=2.0)
    except Exception:
        return None
