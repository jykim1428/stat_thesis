import re, pathlib, markdown

src = pathlib.Path('paper/졸업논문_통합초안_공유용.md').read_text(encoding='utf-8')

# ---- 1. strip editorial scaffolding -------------------------------------
# drop everything from the trailing "통합 시 확인할 것" section
src = src.split('\n# 통합 시 확인할 것')[0].rstrip()
# drop the standalone 목차 block (Docs builds its own TOC)
src = re.sub(r'\n# 목차\n.*?\n---\n', '\n', src, flags=re.S)


# ---- 1b. rebuild cover as pure HTML (Markdown skips ** inside raw div) ---
COVER_HTML = """<p style="text-align:center;font-size:13pt;font-weight:bold;margin-top:36pt;line-height:1.7">
█TITLE█
</p>
<p style="text-align:center;margin-top:120pt;font-size:12pt">한국외국어대학교 통계학과</p>
<p style="text-align:center;font-size:12pt">000</p>
"""
m = re.search(r'<div align="center">(.*?)</div>', src, flags=re.S)
if m:
    inner = m.group(1)
    tm = re.search(r'\*\*(.+?)\*\*', inner, flags=re.S)
    title = tm.group(1).strip() if tm else ''
    title = title.replace('<br>', '<br>\n')
    src = src[:m.start()] + COVER_HTML.replace('█TITLE█', title) + src[m.end():]

lines = src.split('\n')
out, i = [], 0
while i < len(lines):
    L = lines[i]
    # drop author-attribution and typesetting-memo blockquotes
    if L.startswith('>') and re.search(
        r'집필:|표지 배치|수정 이력|유지 담당|초록은|표 4-10이|부록 A\(실험 환경\)', L):
        while i < len(lines) and (lines[i].startswith('>') or lines[i].strip()==''):
            if lines[i].strip()=='' and not (i+1 < len(lines) and lines[i+1].startswith('>')):
                break
            i += 1
        continue
    out.append(L); i += 1
src = '\n'.join(out)
src = re.sub(r'\n{3,}', '\n\n', src)

# ---- 2. math -> readable text (Docs has no LaTeX) ----------------------
# ORDER MATTERS: longer commands first, else \left -> "≤ft" via \le
GREEK = [(r'\mathbb{R}','ℝ'), (r'\qquad','   '), (r'\quad','  '),
         (r'\left',''), (r'\right',''), (r'\lvert','|'), (r'\rvert','|'),
         (r'\gamma','γ'), (r'\lambda','λ'), (r'\sigma','σ'), (r'\mu','μ'),
         (r'\alpha','α'), (r'\Delta','Δ'), (r'\delta','δ'),
         (r'\tilde',''), (r'\bar',''), (r'\hat',''),
         (r'\times','×'), (r'\approx','≈'), (r'\infty','∞'), (r'\pm','±'),
         (r'\cdot','·'), (r'\sum','Σ'), (r'\prod','∏'), (r'\sqrt','√'),
         (r'\min','min'), (r'\max','max'), (r'\log','log'), (r'\exp','exp'),
         (r'\text',''), (r'\in','∈'),
         (r'\leq','≤'), (r'\geq','≥'), (r'\neq','≠'),
         (r'\le','≤'), (r'\ge','≥'), (r'\ne','≠'),
         (r'\;',' '), (r'\,',' ')]
SUB = str.maketrans('0123456789t-+=()ijsnbTW', '₀₁₂₃₄₅₆₇₈₉ₜ₋₊₌₍₎ᵢⱼₛₙᵦₜw')

def _frac(s):
    """Resolve \\frac{A}{B} / \\dfrac{A}{B} with real brace matching."""
    def grab(txt, j):
        # txt[j] must be '{'; return (inner, index_after_closing)
        if j >= len(txt) or txt[j] != '{':
            return None, None
        d = 0
        for k in range(j, len(txt)):
            if txt[k] == '{':
                d += 1
            elif txt[k] == '}':
                d -= 1
                if d == 0:
                    return txt[j + 1:k], k + 1
        return None, None

    pat = re.compile(r'\\dfrac\{|\\frac\{')
    for _ in range(40):
        m = pat.search(s)
        if not m:
            break
        br = m.end() - 1
        a, j = grab(s, br)
        if a is None:
            s = s[:m.start()] + s[m.end():]
            continue
        b, k = grab(s, j)
        if b is None:
            s = s[:m.start()] + '(' + a + ')' + s[j:]
            continue
        s = s[:m.start()] + '(' + a + ')/(' + b + ')' + s[k:]
    return s

def demath(s):
    for k, v in GREEK:
        s = s.replace(k, v)
    # flatten sub/superscript braces FIRST so \frac's arguments become brace-free
    for _ in range(8):
        s2 = re.sub(r'\^\{([^{}]+)\}', lambda m: '^'+m.group(1), s)
        s2 = re.sub(r'_\{([^{}]+)\}', lambda m: '_'+m.group(1), s2)
        if s2 == s:
            break
        s = s2
    s = _frac(s)
    s = s.replace('{','').replace('}','').replace('\\','')
    return re.sub(r'\s+', ' ', s).strip()

# display math -> centered italic paragraph
def disp(m):
    return '\n<p class="eq">' + demath(m.group(1)) + '</p>\n'
src = re.sub(r'\$\$(.+?)\$\$', disp, src, flags=re.S)
src = re.sub(r'\$([^$\n]+?)\$', lambda m: demath(m.group(1)), src)

# ---- 3. footnotes inline ------------------------------------------------
notes = dict(re.findall(r'^\[\^(\w+)\]:\s*(.+?)(?=\n\n|\n\[\^|\Z)', src, flags=re.S|re.M))
src = re.sub(r'^\[\^\w+\]:\s*.+?(?=\n\n|\n\[\^|\Z)', '', src, flags=re.S|re.M)
for k, v in notes.items():
    src = src.replace(f'[^{k}]', f' (주: {" ".join(v.split())})')

html_body = markdown.markdown(src, extensions=['tables', 'sane_lists'])

# callout blockquotes -> boxed divs
html_body = html_body.replace('<blockquote>', '<blockquote class="note">')

doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>졸업논문 통합초안</title>
<style>
body {{ font-family: 'Batang','Times New Roman',serif; font-size: 11pt; line-height: 1.8; }}
h1 {{ font-size: 18pt; font-weight: bold; margin-top: 28pt; }}
h2 {{ font-size: 14pt; font-weight: bold; margin-top: 22pt; }}
h3 {{ font-size: 12pt; font-weight: bold; margin-top: 18pt; }}
p {{ margin: 6pt 0; text-align: justify; }}
table {{ border-collapse: collapse; margin: 10pt 0; font-size: 10pt; }}
th, td {{ border: 1px solid #666; padding: 4pt 7pt; }}
th {{ background: #eee; font-weight: bold; }}
blockquote.note {{ border-left: 3px solid #999; margin: 10pt 0; padding: 2pt 12pt;
                   background: #f6f6f6; font-size: 10.5pt; }}
p.eq {{ text-align: center; font-style: italic; margin: 10pt 0; }}
code {{ font-family: Consolas, monospace; font-size: 10pt; }}
.cover {{ text-align: center; }}
</style></head><body>
{html_body}
</body></html>"""

pathlib.Path('paper/졸업논문_구글독스용_공유용.html').write_text(doc, encoding='utf-8')
print("html chars:", len(doc))
print("tables:", doc.count('<table>'), "| h1:", doc.count('<h1>'), "| h2:", doc.count('<h2>'))
print("residual $:", doc.count('$'), "| residual backslash-cmd:", len(re.findall(r'\\[a-zA-Z]+', doc)))
