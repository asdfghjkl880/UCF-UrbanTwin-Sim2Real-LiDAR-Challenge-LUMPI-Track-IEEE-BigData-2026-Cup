"""Markdown -> Print-ready IEEE-style HTML -> PDF (Edge headless, zero external network dependencies).

Usage:
  D:\app\anaconda\envs\lidar310\python.exe scripts/md2pdf.py --md reports/report_final.md --out reports/report_final.pdf
"""
import argparse
import os
import re
import subprocess
import sys

import markdown

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: letter; margin: 13.0mm 13.5mm 13.0mm 13.5mm; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", Times, Georgia, serif;
  font-size: 10pt; line-height: 1.22; color: #000000; margin: 0;
  text-align: justify;
  column-count: 2; column-gap: 5.5mm;
}
.title-block {
  column-span: all; text-align: center; margin-bottom: 3.2mm;
  padding-bottom: 2.0mm; border-bottom: 0.6pt solid #94a3b8;
}
.title-block h1 {
  font-family: "Times New Roman", Times, serif;
  font-size: 16.0pt; line-height: 1.18; margin: 0 0 1.5mm; color: #000000;
  font-weight: 700;
}
.title-block .subtitle {
  font-family: "Times New Roman", Times, serif;
  font-size: 10.2pt; font-style: italic; color: #1e293b; margin-bottom: 1.0mm;
}
.title-block .author-line {
  font-family: "Times New Roman", Times, serif;
  font-size: 9.6pt; color: #0f172a;
}
.abstract-box {
  column-span: all; font-family: "Times New Roman", Times, serif;
  font-size: 9.0pt; line-height: 1.18; margin: 0 0 3.2mm;
  padding: 2.0mm 3.2mm; background: #f8fafc; border: 0.5pt solid #cbd5e1;
  text-align: justify;
}
.abstract-box p { margin: 0 0 1.0mm; }
.abstract-box p.keywords { margin: 0; }
h2 {
  font-family: "Times New Roman", Times, serif;
  font-size: 10pt; line-height: 1.20; margin: 2.5mm 0 1.0mm; color: #000000;
  text-align: center; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.25px; page-break-after: avoid; break-after: avoid;
}
h3 {
  font-family: "Times New Roman", Times, serif;
  font-size: 10pt; line-height: 1.20; margin: 1.9mm 0 0.7mm; color: #000000;
  font-style: italic; font-weight: 700; page-break-after: avoid; break-after: avoid;
}
h4 {
  font-family: "Times New Roman", Times, serif;
  font-size: 9.5pt; line-height: 1.18; margin: 1.6mm 0 0.6mm; color: #000000;
  font-style: italic; font-weight: 600; page-break-after: avoid; break-after: avoid;
}
p { margin: 0 0 1.3mm; }
.wide-block {
  column-span: all; width: 100%; margin: 1.5mm 0 2.0mm;
  page-break-inside: avoid; break-inside: avoid;
}
.col-block {
  width: 100%; margin: 1.2mm 0 1.6mm;
  page-break-inside: avoid; break-inside: avoid;
}
.table-caption {
  text-align: center; font-family: "Times New Roman", Times, serif;
  font-size: 8.0pt; line-height: 1.18; color: #000000;
  margin: 0 0 0.7mm; text-transform: none;
  page-break-after: avoid; break-after: avoid;
}
.table-caption b {
  font-size: 8.0pt; letter-spacing: 0.4px; text-transform: uppercase;
}
table {
  border-collapse: collapse; width: 100%; table-layout: auto;
  margin: 0; font-family: "Times New Roman", Times, serif;
  font-size: 7.8pt; line-height: 1.18; page-break-inside: avoid; break-inside: avoid;
}
th {
  background: #f1f5f9; border-top: 1.1pt solid #000000; border-bottom: 0.7pt solid #000000;
  border-left: 0.4pt solid #cbd5e1; border-right: 0.4pt solid #cbd5e1;
  padding: 0.65mm 1.0mm; text-align: left; font-weight: 700; color: #000000;
}
td {
  border-top: 0.4pt solid #cbd5e1; border-bottom: 0.4pt solid #cbd5e1;
  border-left: 0.4pt solid #cbd5e1; border-right: 0.4pt solid #cbd5e1;
  padding: 0.55mm 1.0mm; vertical-align: top;
  word-break: normal; overflow-wrap: break-word;
}
tr:last-child td { border-bottom: 1.0pt solid #000000; }
code {
  font-family: Consolas, "Courier New", monospace; font-size: 8.4pt;
  background: #f1f5f9; padding: 0.05mm 0.5mm; border-radius: 1.5px; color: #0f172a;
}
pre {
  background: #f8fafc; border: 0.5pt solid #94a3b8;
  border-left: 2pt solid #0b2545; padding: 1.2mm 1.6mm;
  overflow-x: hidden;
  white-space: pre-wrap; overflow-wrap: anywhere;
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.2pt; line-height: 1.16; margin: 0.8mm 0 1.2mm;
}
pre code { background: none; padding: 0; font-size: 7.2pt; }
ul, ol { margin: 0 0 1.3mm; padding-left: 4.4mm; }
li { margin-bottom: 0.45mm; }
strong { color: #000000; font-weight: 700; }
sub, sup { font-size: 72%; line-height: 0; }
.math-block {
  display: block; text-align: center; margin: 1.2mm 0; padding: 0.6mm 1.0mm;
  background: #f8fafc; border-radius: 2px; font-family: "Cambria Math", "Times New Roman", serif;
  font-size: 9.2pt; line-height: 1.18; font-style: italic;
  overflow-x: hidden;
}
.math-inline {
  font-family: "Cambria Math", "Times New Roman", serif;
  font-style: italic;
}
.math-inline b, .math-block b, .math-rm { font-style: normal; }
.ieee-refs {
  font-family: "Times New Roman", Times, serif;
  font-size: 8.2pt; line-height: 1.15; text-align: justify;
}
.ieee-refs p {
  margin: 0 0 0.55mm; padding-left: 5.0mm; text-indent: -5.0mm;
}
.ieee-refs .ref-num {
  display: inline-block; width: 4.8mm; text-indent: 0; font-weight: normal;
}
"""

LATEX_MAP = [
    (r"\\%", "%"),
    (r"\^\\circ", "°"),
    (r"\\circ", "°"),
    (r"\\times", " × "),
    (r"\\cdot", " · "),
    (r"\\sim", " ~ "),
    (r"\\quad", " &emsp; "),
    (r"\\le\b", " ≤ "),
    (r"\\ge\b", " ≥ "),
    (r"\\ne\b", " ≠ "),
    (r"\\approx", " ≈ "),
    (r"\\equiv", " ≡ "),
    (r"\\propto", " ∝ "),
    (r"\\in\b", " ∈ "),
    (r"\\land\b", " ∧ "),
    (r"\\rightarrow", " → "),
    (r"\\leftarrow", " ← "),
    (r"\\pm", "±"),
    (r"\\infty", "∞"),
    (r"\\dots", "…"),
    (r"\\ldots", "…"),
    (r"\\sum", "∑"),
    (r"\\prod", "∏"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\gamma", "γ"),
    (r"\\delta", "δ"),
    (r"\\Delta", "Δ"),
    (r"\\epsilon", "ε"),
    (r"\\eta", "η"),
    (r"\\theta", "θ"),
    (r"\\lambda", "λ"),
    (r"\\mu", "μ"),
    (r"\\pi", "π"),
    (r"\\rho", "ρ"),
    (r"\\sigma", "σ"),
    (r"\\Sigma", "Σ"),
    (r"\\tau", "τ"),
    (r"\\odot", "⊙"),
    (r"\\quad", " &nbsp;&nbsp; "),
    (r"\\lvert", "&#124;"),
    (r"\\rvert", "&#124;"),
    (r"\\mid", " &#124; "),
    (r"\\;", " "),
    (r"\\,", " "),
    (r"\\!", ""),
    (r"\\big\[", "["),
    (r"\\big\]", "]"),
    (r"\\big\(", "("),
    (r"\\big\)", ")"),
    (r"\\left\(", "("),
    (r"\\right\)", ")"),
    (r"\\left\[", "["),
    (r"\\right\]", "]"),
    (r"\\\{", "{"),
    (r"\\\}", "}"),
    (r"\\\|", "‖"),
]


def _replace_frac(s: str) -> str:
    while r"\frac{" in s:
        idx = s.find(r"\frac{")
        # find matching brace for numerator
        i = idx + 6
        depth = 1
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        num = s[idx + 6 : i - 1]
        if i < len(s) and s[i] == "{":
            j = i + 1
            depth = 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            den = s[i + 1 : j - 1]
            s = s[:idx] + f"({num})/({den})" + s[j:]
        else:
            break
    return s


def latex_to_html(expr: str) -> str:
    s = expr.strip()
    s = _replace_frac(s)
    for pat, rep in LATEX_MAP:
        s = re.sub(pat, rep, s)
    # \text{...}, \mathcal{...}, \mathbb{...}, \mathbf{...}, \boldsymbol{...}
    s = re.sub(r"\\text\{([^{}]*)\}", r'<span class="math-rm">\1</span>', s)
    s = re.sub(r"\\mathcal\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\boldsymbol\{([^{}]*)\}", r"<b>\1</b>", s)
    # \bar{x} -> x̄, \tilde{x} -> x̃
    s = re.sub(r"\\bar\{([^{}]*)\}", r"\1&#772;", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", r"\1&#771;", s)
    # \sqrt{...}
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s = s.replace(r"\top", "⊤")
    # Subscripts and superscripts: _{...} and ^{...} (up to 2 levels of nesting)
    for _ in range(2):
        s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([a-zA-Z0-9α-ωΑ-Ω*])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([a-zA-Z0-9α-ωΑ-Ω⊤*+-])", r"<sup>\1</sup>", s)
    s = s.replace(r"\max", '<span class="math-rm">max</span>')
    s = s.replace(r"\min", '<span class="math-rm">min</span>')
    s = s.replace(r"\exp", '<span class="math-rm">exp</span>')
    s = s.replace(r"\log", '<span class="math-rm">log</span>')
    s = s.replace(r"\sin", '<span class="math-rm">sin</span>')
    s = s.replace(r"\cos", '<span class="math-rm">cos</span>')
    s = s.replace(r"\tan", '<span class="math-rm">tan</span>')
    s = s.replace(r"\tanh", '<span class="math-rm">tanh</span>')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="reports/report_final.md")
    ap.add_argument("--out", default="reports/report_final.pdf")
    ap.add_argument("--html", default="")
    a = ap.parse_args()

    src = open(a.md, encoding="utf-8").read()

    # Convert display math $$...$$ and inline math $...$ into clean HTML spans
    src = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: f'<div class="math-block">{latex_to_html(m.group(1))}</div>',
        src,
        flags=re.S,
    )
    src = re.sub(
        r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
        lambda m: f'<span class="math-inline">{latex_to_html(m.group(1))}</span>',
        src,
    )

    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )

    def _wrap_table(m):
        cap = m.group(1)
        tbl = m.group(2)
        if re.search(r"<b>TABLE (II|III|IV|VI)</b>", cap):
            return f'<div class="wide-block">{cap}{tbl}</div>'
        return f'<div class="col-block">{cap}{tbl}</div>'

    body = re.sub(
        r'(<div class="table-caption">.*?</div>)\s*(<table>.*?</table>)',
        _wrap_table,
        body,
        flags=re.S,
    )
    html = (
        f'<!DOCTYPE html><html lang="en-US"><head><meta charset="utf-8">'
        f"<title>UCF UrbanTwin Sim2Real LiDAR Challenge — LUMPI Track Technical Report</title>"
        f"<style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )

    hp = a.html or os.path.splitext(a.out)[0] + ".html"
    open(hp, "w", encoding="utf-8").write(html)
    print(f"HTML -> {hp}  ({len(html)/1024:.1f} KB)")

    url = "file:///" + os.path.abspath(hp).replace("\\", "/")
    out = os.path.abspath(a.out)
    tmp_dir = r"D:\app\anaconda\cache\tmp\edge_pdf"
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        EDGE,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={tmp_dir}",
        f"--print-to-pdf={out}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out):
        print("STDOUT", r.stdout[-800:])
        print("STDERR", r.stderr[-800:])
        sys.exit("PDF generation failed")
    print(f"PDF  -> {out}  ({os.path.getsize(out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
