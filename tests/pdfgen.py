"""Builds genuine, text-based PDF files (real content streams + xref table) for extraction tests."""
from __future__ import annotations


def _esc(text: str) -> str:
    out = []
    for ch in text:
        if ch in "()\\":
            out.append("\\" + ch)
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("\\%03o" % ch.encode("cp1252")[0])   # WinAnsi octal escape (Latin-1 range)
    return "".join(out)


def make_pdf(pages: list[str]) -> bytes:
    """Each page is a string; newlines become separate text lines. Uses Helvetica / WinAnsiEncoding."""
    objs: list[bytes] = []                       # object i+1
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(("<< /Type /Pages /Kids [" + " ".join(f"{p} 0 R" for p in page_ids) + f"] /Count {n} >>").encode())
    for i, text in enumerate(pages):
        lines = text.split("\n")
        stream = "BT /F1 12 Tf 50 740 Td 16 TL\n" + "\n".join(f"({_esc(l)}) Tj T*" for l in lines) + "\nET"
        sb = stream.encode("latin-1")
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_ids[i] + 1} 0 R "
                     f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>").encode())
        objs.append(f"<< /Length {len(sb)} >>\nstream\n".encode() + sb + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
