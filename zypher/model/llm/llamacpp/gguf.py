"""Read the header of a GGUF file: just enough to size the context window.

What a context token costs in VRAM depends on the model's shape -- layers,
KV heads, head width -- and all of that is in the file's metadata. Reading it
here means the memory plan is right for whatever model is dropped in, rather
than for the one 7B it was first worked out for.
"""

import struct

# GGUF value types that are a fixed-size scalar: type id -> struct format.
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING = 8
_ARRAY = 9

# Bytes per stored value for each KV cache type llama.cpp offers. The
# quantized ones carry a scale per block of 32 on top of the values.
KV_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}


def _read(handle, fmt):
    size = struct.calcsize(fmt)
    data = handle.read(size)

    if len(data) != size:
        raise ValueError("unexpected end of GGUF header")

    return struct.unpack(fmt, data)[0]


def _read_string(handle):
    return handle.read(_read(handle, "<Q")).decode("utf-8", "replace")


def _read_value(handle, kind):
    if kind in _SCALARS:
        return _read(handle, _SCALARS[kind])

    if kind == _STRING:
        return _read_string(handle)

    if kind == _ARRAY:
        item = _read(handle, "<I")
        count = _read(handle, "<Q")

        # Arrays are the vocabulary and its scores: megabytes nobody here
        # needs. Skipped, not read.
        if item in _SCALARS:
            handle.seek(count * struct.calcsize(_SCALARS[item]), 1)
        elif item == _STRING:
            for _ in range(count):
                handle.seek(_read(handle, "<Q"), 1)
        else:
            raise ValueError("nested arrays are not supported")

        return None

    raise ValueError("unknown GGUF value type {}".format(kind))


def read_metadata(path):
    """Scalar and string metadata of a GGUF file, as a dict."""

    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("{} is not a GGUF file".format(path))

        _read(handle, "<I")  # version
        _read(handle, "<Q")  # tensor count
        pairs = _read(handle, "<Q")

        meta = {}

        for _ in range(pairs):
            key = _read_string(handle)
            value = _read_value(handle, _read(handle, "<I"))

            if value is not None:
                meta[key] = value

    return meta


def describe(path):
    """The few numbers the memory plan needs, with safe fallbacks.

    Returns name, context (the window the model was trained with) and
    kv_values (stored values per context token, before the cache's dtype).
    """

    meta = read_metadata(path)
    arch = meta.get("general.architecture", "llama")

    def field(name, default=None):
        return meta.get("{}.{}".format(arch, name), default)

    layers = field("block_count", 32)
    width = field("embedding_length", 4096)
    heads = field("attention.head_count", 32) or 32
    kv_heads = field("attention.head_count_kv", heads) or heads
    key_len = field("attention.key_length", width // heads)
    value_len = field("attention.value_length", width // heads)

    return {
        "name": meta.get("general.name") or arch,
        "context": int(field("context_length", 4096)),
        "kv_values": int(layers * kv_heads * (key_len + value_len)),
    }


def kv_bytes_per_token(shape, kv_type):
    return shape["kv_values"] * KV_BYTES.get(kv_type, 2.0)
