import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "negpy"


def _is_jit(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
    return name in ("njit", "jit", "parallel_njit")


def _inline_only(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and _is_jit(decorator)
        and any(k.arg == "inline" and getattr(k.value, "value", None) == "always" for k in decorator.keywords)
    )


def _module(path: Path) -> str:
    return ".".join(path.relative_to(_ROOT.parent).with_suffix("").parts)


def test_python_code_never_calls_an_uncached_inline_kernel():
    """An inline helper has no cache, so a call from plain Python compiles it again on every launch."""
    trees = {_module(p): ast.parse(p.read_text(encoding="utf-8")) for p in _ROOT.rglob("*.py")}
    inline = {
        mod: {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and any(_inline_only(d) for d in n.decorator_list)}
        for mod, tree in trees.items()
    }
    offenders = []
    for mod, tree in trees.items():
        visible = set(inline[mod])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in inline:
                visible |= {a.asname or a.name for a in node.names if a.name in inline[node.module]}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or any(_is_jit(d) for d in fn.decorator_list):
                continue
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) in visible:
                    offenders.append(f"{mod}:{call.lineno} {call.func.id}")
    assert offenders == []
