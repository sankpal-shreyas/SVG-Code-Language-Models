"""SVG cleaning. Parse with lxml, drop noise, round coords to 1 decimal place."""
import re
from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
DROP_TAGS = {"title", "desc", "metadata", "defs"}
NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _round_numbers(text: str, ndigits: int = 1) -> str:
    def repl(m):
        try:
            v = float(m.group(0))
        except ValueError:
            return m.group(0)
        r = round(v, ndigits)
        if r == int(r):
            return f"{int(r)}"
        return f"{r:g}"
    return NUMBER_RE.sub(repl, text)


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _walk_drop(elem):
    for child in list(elem):
        local = _strip_namespace(child.tag) if isinstance(child.tag, str) else ""
        if local in DROP_TAGS:
            elem.remove(child)
        else:
            _walk_drop(child)


def normalize(svg_str: str, ndigits: int = 1) -> str | None:
    """Return cleaned SVG string, or None if it cannot be parsed."""
    if not svg_str or "<svg" not in svg_str:
        return None
    try:
        parser = etree.XMLParser(remove_comments=True, remove_blank_text=True, recover=False)
        root = etree.fromstring(svg_str.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None

    if not isinstance(root.tag, str) or _strip_namespace(root.tag) != "svg":
        return None

    _walk_drop(root)

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        new_attrib = {}
        for k, v in elem.attrib.items():
            local = _strip_namespace(k)
            if local in {"id", "data-name", "class"} and len(v) > 16:
                continue
            new_attrib[local] = _round_numbers(v, ndigits)
        elem.attrib.clear()
        for k, v in new_attrib.items():
            elem.set(k, v)
        if elem.text is not None:
            elem.text = elem.text.strip() or None
        if elem.tail is not None:
            elem.tail = elem.tail.strip() or None

    out = etree.tostring(root, encoding="unicode")
    out = re.sub(r"\s+", " ", out).strip()
    if "xmlns" not in out:
        out = out.replace("<svg ", f'<svg xmlns="{SVG_NS}" ', 1) if "<svg " in out else out.replace("<svg>", f'<svg xmlns="{SVG_NS}">', 1)
    return out


def is_valid(svg_str: str) -> bool:
    try:
        etree.fromstring(svg_str.encode("utf-8"), etree.XMLParser(recover=False))
        return True
    except Exception:
        return False
