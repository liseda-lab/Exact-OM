import json, re, sys
css = open(sys.argv[1]).read()
def block(selector):
    i = css.index(selector); j = css.index("}", i)
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", css[i:j]))
light = block(":root {"); dark = {**light, **block(':root[data-theme="dark"] {')}
def rgb(h): h = h.strip().lstrip("#"); return tuple(int(h[k:k+2], 16) for k in (0, 2, 4))
def lum(c):
    def ch(v):
        v /= 255; return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c; return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)
def ratio(a, b):
    x, y = sorted([lum(a), lum(b)], reverse=True); return round((x + 0.05) / (y + 0.05), 2)
def mix(fg, bg, alpha): return tuple(round(alpha * f + (1 - alpha) * b) for f, b in zip(fg, bg))
rows = []
for theme, t in (("light", light), ("dark", dark)):
    c = {k: rgb(t[k]) for k in ("--ink", "--ink-2", "--ink-3", "--surface", "--surface-2")}
    cases = [
        ("before S0 follow-up: faded focal label (ink at opacity 0.35 over surface-2)", mix(c["--ink"], c["--surface-2"], 0.35), c["--surface-2"]),
        ("before S0 follow-up: faded feature label (ink at opacity 0.35 over surface)", mix(c["--ink"], c["--surface"], 0.35), c["--surface"]),
        ("focal label, any state (ink on surface-2)", c["--ink"], c["--surface-2"]),
        ("entity/literal label (ink on surface)", c["--ink"], c["--surface"]),
        ("faded entity/literal label (ink-2 on surface)", c["--ink-2"], c["--surface"]),
        ("relation label on a line (ink-2 on its surface background)", c["--ink-2"], c["--surface"]),
        ("selected relation label (ink on surface)", c["--ink"], c["--surface"]),
        ("group count label (ink-3 on surface)", c["--ink-3"], c["--surface"]),
    ]
    for name, fg, bg in cases:
        rows.append({"theme": theme, "case": name, "text": "#%02x%02x%02x" % fg, "background": "#%02x%02x%02x" % bg, "contrast": ratio(fg, bg), "required": 4.5, "meets": ratio(fg, bg) >= 4.5})
print(json.dumps(rows, indent=1))
