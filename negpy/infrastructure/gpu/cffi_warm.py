"""Pre-fills wgpu-py's cffi type cache from the models cffi already built for webgpu.h.

wgpu-py runs cffi in ABI mode, so the first ``ffi.new("WGPUFoo *")`` for every distinct type
string goes through pycparser, and each parse re-declares every typedef of the header. Dozens of
those land on the GUI thread during window construction and on the first render. The typedef
models are already in the parser's declarations, so the backend types can be built directly.
Private cffi API: any failure leaves the cache untouched and wgpu parses as before.
"""

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_PRIMITIVES = (
    "char",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int32_t",
    "int64_t",
    "float",
    "double",
    "size_t",
    "int",
    "uintptr_t",
    "intptr_t",
)
_warmed = False


def warm_wgpu_cffi_types() -> int:
    """Returns the number of cache entries added; 0 when already warmed or unavailable."""
    global _warmed
    if _warmed:
        return 0
    _warmed = True
    try:
        import cffi.model as cm
        from wgpu.backends.wgpu_native._ffi import ffi

        added = 0
        # _get_cached_btype asserts the ffi lock is held; without it the assert's
        # acquire(False) leaks the lock and the next real parse deadlocks.
        with ffi._lock:

            def put(cdecl: str, model) -> None:
                nonlocal added
                if cdecl not in ffi._parsed_types:
                    ffi._parsed_types[cdecl] = (ffi._get_cached_btype(model), False)
                    added += 1

            for key, val in list(ffi._parser._declarations.items()):
                if not key.startswith("typedef "):
                    continue
                model = val[0] if isinstance(val, tuple) else val
                if not isinstance(model, (cm.StructType, cm.EnumType, cm.PrimitiveType, cm.PointerType)):
                    continue
                name = key[len("typedef ") :]
                put(name, model)
                put(name + " *", cm.PointerType(model))
                put(name + "[]", cm.ArrayType(model, None))
            for prim in _PRIMITIVES:
                model = cm.PrimitiveType(prim)
                put(prim, model)
                put(prim + " *", cm.PointerType(model))
                put(prim + "[]", cm.ArrayType(model, None))
                put(prim + " []", cm.ArrayType(model, None))
            put("void *", cm.PointerType(cm.void_type))
        return added
    except Exception as e:
        logger.debug("cffi type warm skipped: %s", e)
        return 0
