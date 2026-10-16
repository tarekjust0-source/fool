"""Pure-Python/numpy image utilities.

Everything here is independent of ``bpy`` so it can run inside worker threads
and in plain unit tests.  Arrays follow the usual image convention: shape
``(H, W, C)`` with row 0 at the *top*.  (Blender's pixel buffers start at the
bottom row; the Blender bridge flips when converting.)
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Hard limits protecting against decompression bombs / absurd responses.
MAX_DECODE_PIXELS = 8192 * 8192
MAX_ENCODED_BYTES = 128 * 1024 * 1024


class ImageDecodeError(ValueError):
    """Raised when image bytes cannot be decoded by the pure-Python decoder."""


class InvalidImageError(ValueError):
    """Raised when bytes decode but the content is not a usable image."""


# ---------------------------------------------------------------------------
# Format sniffing / validation
# ---------------------------------------------------------------------------

def sniff_format(data: bytes) -> str | None:
    """Return 'png', 'jpeg', 'webp', 'gif', 'bmp' or None from magic bytes."""
    if not data or len(data) < 12:
        return None
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    return None


def validate_image_array(arr: np.ndarray, *, min_size: int = 16, reject_uniform: bool = True) -> np.ndarray:
    """Validate a decoded image and return it as uint8 RGB/RGBA.

    Rejects tiny images and (optionally) images that are a single flat colour,
    which in practice means the provider returned a blank/failed result.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
        raise InvalidImageError("Decoded data is not an image array")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    if h < min_size or w < min_size:
        raise InvalidImageError(f"Image is too small ({w}x{h})")
    if c not in (1, 2, 3, 4):
        raise InvalidImageError(f"Unsupported channel count {c}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c == 2:
        arr = np.concatenate([np.repeat(arr[:, :, :1], 3, axis=2), arr[:, :, 1:2]], axis=2)
    if reject_uniform:
        rgb = arr[:, :, :3]
        # Sample to keep this cheap on big images.
        step = max(1, min(h, w) // 256)
        sample = rgb[::step, ::step].reshape(-1, 3).astype(np.int16)
        if int(sample.max()) - int(sample.min()) <= 2:
            raise InvalidImageError("Provider returned a blank (single-colour) image")
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------

def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_png(arr: np.ndarray, level: int = 6) -> bytes:
    """Encode a uint8 (H,W), (H,W,1..4) array as PNG.

    Uses the PNG "Up" filter on every row: decodes fast with vectorised numpy
    and compresses textures reasonably well.
    """
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(c)
    if color_type is None:
        raise ValueError(f"Cannot encode {c}-channel image as PNG")
    arr = np.ascontiguousarray(arr)
    flat = arr.reshape(h, w * c)
    raw = np.empty((h, w * c + 1), dtype=np.uint8)
    raw[:, 0] = 2  # Up filter
    raw[0, 1:] = flat[0]
    if h > 1:
        raw[1:, 1:] = flat[1:] - flat[:-1]  # uint8 arithmetic wraps mod 256
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(raw.tobytes(), level)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# PNG decoding
# ---------------------------------------------------------------------------

def _paeth(a, b, c):
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _unfilter(raw: np.ndarray, bpp: int) -> np.ndarray:
    """Undo PNG scanline filters. ``raw`` has shape (H, 1 + stride)."""
    h = raw.shape[0]
    ftypes = raw[:, 0]
    data = raw[:, 1:]
    stride = data.shape[1]
    if np.any(ftypes > 4):
        raise ImageDecodeError("Invalid PNG filter type")
    out = np.empty((h, stride), dtype=np.uint8)

    if not np.any((ftypes == 3) | (ftypes == 4)):
        # Fast path: None / Sub / Up are vectorisable per row.
        prev = np.zeros(stride, dtype=np.uint8)
        for r in range(h):
            t = ftypes[r]
            line = data[r]
            if t == 0:
                cur = line
            elif t == 2:
                cur = line + prev
            else:  # Sub: cumulative sum per byte-lane, mod 256
                if stride % bpp:
                    pad = bpp - stride % bpp
                    tmp = np.concatenate([line, np.zeros(pad, np.uint8)]).reshape(-1, bpp)
                    cur = np.cumsum(tmp, axis=0, dtype=np.uint8).reshape(-1)[:stride]
                else:
                    cur = np.cumsum(line.reshape(-1, bpp), axis=0, dtype=np.uint8).reshape(-1)
            out[r] = cur
            prev = out[r]
        return out

    # General path (Average / Paeth present): anti-diagonal wavefront so each
    # step is vectorised over all pixels that only depend on finished ones.
    if stride % bpp:
        pad = bpp - stride % bpp
        data = np.concatenate([data, np.zeros((h, pad), np.uint8)], axis=1)
    wp = data.shape[1] // bpp
    filt = data.reshape(h, wp, bpp).astype(np.int16)
    rec = np.zeros((h, wp, bpp), dtype=np.int16)
    ft_all = ftypes.astype(np.int16)
    for k in range(h + wp - 1):
        r0 = max(0, k - wp + 1)
        r1 = min(h - 1, k)
        rows = np.arange(r0, r1 + 1)
        cols = k - rows
        has_left = (cols > 0)[:, None]
        has_up = (rows > 0)[:, None]
        a = np.where(has_left, rec[rows, cols - 1], 0)
        b = np.where(has_up, rec[rows - 1, cols], 0)
        c = np.where(has_left & has_up, rec[rows - 1, cols - 1], 0)
        t = ft_all[rows][:, None]
        pred = np.select(
            [t == 0, t == 1, t == 2, t == 3],
            [np.zeros_like(a), a, b, (a + b) // 2],
            default=_paeth(a, b, c),
        )
        rec[rows, cols] = (filt[rows, cols] + pred) & 0xFF
    return rec.reshape(h, wp * bpp)[:, :stride].astype(np.uint8)


def decode_png(data: bytes, max_pixels: int = MAX_DECODE_PIXELS) -> np.ndarray:
    """Decode PNG bytes into a uint8 array (H, W, 3|4).

    Supports non-interlaced greyscale, RGB, palette, grey+alpha and RGBA at
    bit depths 1/2/4/8/16.  Interlaced PNGs raise ImageDecodeError so callers
    can fall back to Blender's decoder on the main thread.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise ImageDecodeError("Not a PNG file")
    pos = len(PNG_SIGNATURE)
    ihdr = None
    palette = None
    trns = None
    idat = []
    n = len(data)
    while pos + 8 <= n:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc_bytes = data[pos + 8 + length:pos + 12 + length]
        if len(payload) != length or len(crc_bytes) != 4:
            raise ImageDecodeError("Truncated PNG chunk")
        crc = zlib.crc32(tag)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        if struct.unpack(">I", crc_bytes)[0] != crc:
            raise ImageDecodeError(f"PNG CRC mismatch in {tag!r} chunk")
        pos += 12 + length
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"PLTE":
            palette = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat.append(payload)
        elif tag == b"IEND":
            break
    if ihdr is None or not idat:
        raise ImageDecodeError("PNG missing IHDR or IDAT")
    w, h, depth, ctype, comp, filt_method, interlace = ihdr
    if w == 0 or h == 0 or w * h > max_pixels:
        raise ImageDecodeError(f"PNG dimensions {w}x{h} out of range")
    if comp != 0 or filt_method != 0:
        raise ImageDecodeError("Unsupported PNG compression/filter method")
    if interlace != 0:
        raise ImageDecodeError("Interlaced PNG is not supported by the fast decoder")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if channels is None or depth not in (1, 2, 4, 8, 16):
        raise ImageDecodeError("Unsupported PNG colour type / bit depth")
    bits_pp = channels * depth
    stride = (w * bits_pp + 7) // 8
    bpp = max(1, bits_pp // 8)
    expected = h * (stride + 1)
    dobj = zlib.decompressobj()
    try:
        raw = dobj.decompress(b"".join(idat), expected + 1)
    except zlib.error as exc:
        raise ImageDecodeError(f"Corrupt PNG data: {exc}") from exc
    if len(raw) < expected:
        raise ImageDecodeError("PNG image data is truncated")
    raw = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(h, stride + 1)
    rows = _unfilter(raw, bpp)

    if depth == 16:
        px = rows.reshape(h, w, channels, 2)[..., 0]
    elif depth == 8:
        px = rows.reshape(h, w, channels)
    else:
        bits = np.unpackbits(rows, axis=1)[:, : w * depth].reshape(h, w, depth)
        weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
        vals = (bits * weights).sum(axis=2).astype(np.uint8)
        if ctype == 0:
            vals = (vals.astype(np.uint16) * 255 // ((1 << depth) - 1)).astype(np.uint8)
        px = vals[:, :, None]

    if ctype == 3:
        if palette is None:
            raise ImageDecodeError("Palette PNG without PLTE chunk")
        idx = px[:, :, 0]
        if idx.max() >= len(palette):
            raise ImageDecodeError("Palette index out of range")
        rgb = palette[idx]
        if trns is not None:
            alpha_tab = np.full(len(palette), 255, np.uint8)
            t = np.frombuffer(trns, dtype=np.uint8)[: len(palette)]
            alpha_tab[: len(t)] = t
            return np.ascontiguousarray(np.concatenate([rgb, alpha_tab[idx][:, :, None]], axis=2))
        return np.ascontiguousarray(rgb)
    if ctype == 0:
        return np.ascontiguousarray(np.repeat(px, 3, axis=2))
    if ctype == 4:
        g = px[:, :, :1]
        return np.ascontiguousarray(np.concatenate([g, g, g, px[:, :, 1:2]], axis=2))
    return np.ascontiguousarray(px)


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes in a thread-safe way (PNG only).

    Raises ImageDecodeError for formats this module cannot decode; the Blender
    bridge catches that and decodes on the main thread with Blender's
    image loader (JPEG / WebP / interlaced PNG).
    """
    if len(data) > MAX_ENCODED_BYTES:
        raise InvalidImageError("Image payload is unreasonably large")
    fmt = sniff_format(data)
    if fmt is None:
        raise InvalidImageError("Response is not a recognised image format")
    if fmt != "png":
        raise ImageDecodeError(f"{fmt} requires Blender's decoder")
    return decode_png(data)


# ---------------------------------------------------------------------------
# Conversions / resampling
# ---------------------------------------------------------------------------

def to_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    return arr.astype(np.float32, copy=False)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] >= 3:
        return arr[:, :, :3]
    return np.repeat(arr[:, :, :1], 3, axis=2)


def rgba(arr: np.ndarray, alpha=None) -> np.ndarray:
    base = rgb(arr)
    if alpha is None:
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr
        fill = 255 if base.dtype == np.uint8 else 1.0
        alpha = np.full(base.shape[:2], fill, dtype=base.dtype)
    return np.concatenate([base, np.asarray(alpha, dtype=base.dtype)[:, :, None]], axis=2)


def _resample_axis(a: np.ndarray, new_n: int, axis: int) -> np.ndarray:
    n = a.shape[axis]
    if new_n == n:
        return a
    a = np.moveaxis(a, axis, 0)
    if new_n < n:
        # Exact area averaging using a prefix sum with fractional ends.
        cs = np.concatenate([np.zeros((1,) + a.shape[1:], np.float64), np.cumsum(a, axis=0, dtype=np.float64)])
        edges = np.linspace(0.0, n, new_n + 1)
        i = np.floor(edges).astype(np.int64)
        f = edges - i
        i_c = np.minimum(i, n - 1)
        extra = a[i_c].astype(np.float64) * f.reshape((-1,) + (1,) * (a.ndim - 1))
        s = cs[i] + extra
        out = (s[1:] - s[:-1]) / (edges[1:] - edges[:-1]).reshape((-1,) + (1,) * (a.ndim - 1))
        out = out.astype(np.float32)
    else:
        x = (np.arange(new_n) + 0.5) * (n / new_n) - 0.5
        x = np.clip(x, 0, n - 1)
        i0 = np.floor(x).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = (x - i0).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
        out = a[i0] * (1.0 - f) + a[i1] * f
    return np.moveaxis(out, 0, axis)


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with area averaging (down) / bilinear (up). Keeps dtype."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    was_u8 = img.dtype == np.uint8
    was_bool = img.dtype == np.bool_
    f = img.astype(np.float32)
    f = _resample_axis(f, height, 0)
    f = _resample_axis(f, width, 1)
    if was_bool:
        return f >= 0.5
    if was_u8:
        return np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return f.astype(img.dtype, copy=False)


def resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height and w == width:
        return img
    ys = np.minimum((np.arange(height) + 0.5) * h / height, h - 1).astype(np.int64)
    xs = np.minimum((np.arange(width) + 0.5) * w / width, w - 1).astype(np.int64)
    return img[ys][:, xs]


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``img`` (H,W[,C]) at pixel coordinates (pixel centres at +0.5)."""
    h, w = img.shape[:2]
    fx = np.clip(x - 0.5, 0, w - 1)
    fy = np.clip(y - 0.5, 0, h - 1)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (fx - x0).astype(np.float32)
    ay = (fy - y0).astype(np.float32)
    src = img.astype(np.float32, copy=False)
    if src.ndim == 3:
        ax = ax[:, None]
        ay = ay[:, None]
    top = src[y0, x0] * (1 - ax) + src[y0, x1] * ax
    bot = src[y1, x0] * (1 - ax) + src[y1, x1] * ax
    return top * (1 - ay) + bot * ay


def sample_nearest(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xi = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    yi = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    return img[yi, xi]


def luminance(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f[..., 0] * 0.2126 + rgb_f[..., 1] * 0.7152 + rgb_f[..., 2] * 0.0722


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an image by integer pixels, replicating edge pixels."""
    if dy == 0 and dx == 0:
        return img
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    return img[ys][:, xs]
