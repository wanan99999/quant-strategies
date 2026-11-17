# -*- coding: utf-8 -*-
"""生成策略库网站：python build/build.py  →  在仓库根目录写出 index.html、s/*.html、files/**、code/**、docs/**"""
import os, re, shutil, html, zipfile, json, datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import CATS, S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = html.escape
TODAY = datetime.date.today().isoformat()

# ---------------------------------------------------------------- 读文件
def read_text(p):
    b = open(p, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"): return b.decode("utf-16")
    if b[:3] == b"\xef\xbb\xbf": return b[3:].decode("utf8")
    try: return b.decode("utf8")
    except UnicodeDecodeError: return b.decode("gbk", errors="replace")

def human(n):
    return f"{n/1024/1024:.1f} MB" if n >= 1024 * 1024 else f"{max(1, round(n/1024))} KB"

def u(*parts): return "/".join(quote(p) for p in parts)

# ---------------------------------------------------------------- Markdown → HTML（够用的子集）
def inline(s):
    s = E(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    return s

def md2html(t):
    lines = t.replace("\r\n", "\n").replace("\r", "\n").split("\n"); out = []; i = 0
    def is_tbl_sep(l): return re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", l) is not None
    def cells(l):
        l = l.strip()
        if l.startswith("|"): l = l[1:]
        if l.endswith("|"): l = l[:-1]
        return [c.strip() for c in re.split(r"(?<!\\)\|", l)]
    while i < len(lines):
        l = lines[i]
        if l.strip().startswith("```"):
            lang = l.strip()[3:].strip(); buf = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"): buf.append(lines[i]); i += 1
            code = "\n".join(buf)
            out.append(highlight(code, PythonLexer(), HtmlFormatter(cssclass="hl")) if lang.startswith("py") else f"<pre class=\"plain\"><code>{E(code)}</code></pre>")
            i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", l)
        if m:
            n = min(len(m.group(1)) + 1, 6); out.append(f"<h{n}>{inline(m.group(2).strip())}</h{n}>"); i += 1; continue
        if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", l): out.append("<hr>"); i += 1; continue
        if "|" in l and i + 1 < len(lines) and is_tbl_sep(lines[i + 1]):
            head = cells(l); i += 2; rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip(): rows.append(cells(lines[i])); i += 1
            h = "<div class=\"tbl\"><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            h += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
            out.append(h); continue
        if l.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"): buf.append(lines[i].lstrip(">").strip()); i += 1
            out.append("<blockquote>" + "<br>".join(inline(b) for b in buf if b) + "</blockquote>"); continue
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", l)
        if m:
            ordered = m.group(2)[0].isdigit(); tag = "ol" if ordered else "ul"; items = []
            while i < len(lines):
                m2 = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", lines[i])
                if m2: items.append((len(m2.group(1)), m2.group(3))); i += 1
                elif lines[i].strip() and lines[i].startswith("  ") and items: items[-1] = (items[-1][0], items[-1][1] + " " + lines[i].strip()); i += 1
                else: break
            out.append(f"<{tag}>" + "".join(f"<li{' class=sub' if ind >= 2 else ''}>{inline(x)}</li>" for ind, x in items) + f"</{tag}>"); continue
        if not l.strip(): i += 1; continue
        buf = [l.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|>|```|\s*([-*+]|\d+[.)])\s|\s*(-{3,})\s*$)", lines[i]) \
                and not ("|" in lines[i] and i + 1 < len(lines) and is_tbl_sep(lines[i + 1])):
            buf.append(lines[i].strip()); i += 1
        out.append("<p>" + "<br>".join(inline(b) for b in buf) + "</p>")
    return "\n".join(out)

# ---------------------------------------------------------------- Word → HTML（段落、标题、列表、表格、加粗）
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
def docx2html(p):
    z = zipfile.ZipFile(p); doc = ET.fromstring(z.read("word/document.xml"))
    names = {}
    if "word/styles.xml" in z.namelist():
        for s in ET.fromstring(z.read("word/styles.xml")).iter(W + "style"):
            n = s.find(W + "name"); names[s.get(W + "styleId")] = (n.get(W + "val") if n is not None else "").lower()
    def runs(par):
        out = []
        for r in par.iter(W + "r"):
            rp = r.find(W + "rPr"); b = rp is not None and rp.find(W + "b") is not None and rp.find(W + "b").get(W + "val") not in ("0", "false")
            t = ""
            for c in r:
                if c.tag == W + "t": t += c.text or ""
                elif c.tag == W + "tab": t += "　"
                elif c.tag == W + "br": t += "\n"
            if t: out.append((t, b))
        h = ""
        for t, b in out:
            s = E(t).replace("\n", "<br>"); h += f"<strong>{s}</strong>" if b else s
        return h.replace("</strong><strong>", "")
    def para(pp):
        pPr = pp.find(W + "pPr"); sty = ""; lst = False
        if pPr is not None:
            ps = pPr.find(W + "pStyle"); sty = names.get(ps.get(W + "val"), ps.get(W + "val")) if ps is not None else ""
            lst = pPr.find(W + "numPr") is not None
        return sty, lst, runs(pp)
    def blank(h): return not re.sub(r"[\s​　]|<br>", "", h)
    def blocks(container, in_cell=False):
        out = []; ul = []
        def flush():
            if ul: out.append("<ul>" + "".join(f"<li>{x}</li>" for x in ul) + "</ul>"); ul.clear()
        for el in container:
            if el.tag == W + "p":
                sty, lst, h = para(el)
                if sty.startswith("toc") or blank(h): continue
                m = re.match(r"(heading|标题)\s*(\d)", sty)
                if m and not in_cell: flush(); n = min(int(m.group(2)) + 1, 6); out.append(f"<h{n}>{h}</h{n}>"); continue
                if sty == "title" and not in_cell: flush(); out.append(f"<h2>{h}</h2>"); continue
                plain = re.sub(r"<[^>]+>", "", h).strip()
                if not in_cell and re.fullmatch(r"【[^】]{2,40}】", plain): flush(); out.append(f"<h3>{plain}</h3>"); continue
                if lst or "list" in sty: ul.append(h); continue
                flush(); out.append(f"<p>{h}</p>" if not in_cell else h)
            elif el.tag == W + "tbl":
                flush(); trs = el.findall(W + "tr")
                cells = [[tc for tc in tr.findall(W + "tc")] for tr in trs]
                if len(cells) == 1 and len(cells[0]) == 1:              # 1×1 外框表：直接展开内容
                    out.append(blocks(cells[0][0])); continue
                rows = [[blocks(tc, True).replace("\n", "<br>") for tc in r] for r in cells]
                if rows:
                    h = "<div class=\"tbl\"><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr></thead><tbody>"
                    h += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table></div>"
                    out.append(h)
            elif el.tag in (W + "sdt", W + "sdtContent", W + "customXml"):
                inner = el.find(W + "sdtContent") if el.tag == W + "sdt" else el
                if inner is not None: flush(); out.append(blocks(inner, in_cell))
        flush()
        return "\n".join(x for x in out if x)
    return blocks(doc.find(W + "body"))

# ---------------------------------------------------------------- 代码文件头说明
def code_note(t):
    m = re.search(r'"""(.*?)"""', t, re.S)
    if not m: return ""
    ls = [x.strip() for x in m.group(1).strip().splitlines() if x.strip() and not set(x.strip()) <= set("=-")]
    return ls[0] if ls else ""

def code_detail(t):
    m = re.search(r'"""(.*?)"""', t, re.S)
    if not m: return ""
    ls = [x.strip() for x in m.group(1).strip().splitlines() if x.strip() and not set(x.strip()) <= set("=-")]
    for x in ls[1:]:
        if re.match(r"^(V[\d.]+|全板块 V|【本版|对照|在 V)", x): return x
    return ls[1] if len(ls) > 1 else ""

# ---------------------------------------------------------------- 页面框架
def page(title, body, depth=0, desc=""):
    pre = "../" * depth
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)}</title>
<meta name="description" content="{E(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="{pre}assets/site.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232a78d6'/%3E%3Cpath d='M6 22l6-7 5 4 9-11' stroke='white' stroke-width='3' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
</head>
<body>
<header class="top"><div class="bar"><a class="brand" href="{pre}index.html">量化策略库</a>
<nav>{"".join(f'<a href="{pre}index.html#{k}">{n}</a>' for k, n, _ in CATS)}</nav></div></header>
<main class="wrap">
{body}
</main>
<script>
/* 返回键：从本站页面进来就回到上一页（保留原来的滚动位置），直接打开时跳到 href */
document.querySelectorAll("[data-back]").forEach(a => a.addEventListener("click", e => {{
  try {{ if (document.referrer && new URL(document.referrer).origin === location.origin && history.length > 1) {{ e.preventDefault(); history.back(); }} }} catch (err) {{}}
}}));
</script>
<footer class="foot"><div class="wrap-f">本地测算与聚宽回测结果仅供研究参考，不构成投资建议。更新于 {TODAY}</div></footer>
</body>
</html>
"""

ICON = {"pdf": "PDF", "html": "网页", "xlsx": "Excel", "md": "文档", "docx": "文档", "py": "代码"}

def build():
    for d in ("s", "files", "code", "docs"):
        shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    fmt = HtmlFormatter(cssclass="hl", linenos="table")
    light = HtmlFormatter(style="default").get_style_defs(".hl")
    try: dstyle = HtmlFormatter(style="github-dark").get_style_defs(".hl")
    except Exception: dstyle = HtmlFormatter(style="monokai").get_style_defs(".hl")
    dark_scoped = "\n".join(":root:not([data-theme=\"light\"]) " + l if l.startswith(".hl") else l for l in dstyle.splitlines())
    css = open(os.path.join(ROOT, "build", "site.css"), encoding="utf8").read()
    css += "\n/* 代码高亮：浅色 */\n" + light + "\n/* 代码高亮：深色 */\n@media (prefers-color-scheme: dark) {\n" + dark_scoped + "\n}\n"
    open(os.path.join(ROOT, "assets", "site.css"), "w", encoding="utf8").write(css)
    stats = dict(s=len(S), r=0, c=0, d=0)
    cards = {k: [] for k, _, _ in CATS}
    for st in S:
        slug = st["slug"]; fdir = os.path.join(ROOT, "files", slug); os.makedirs(fdir, exist_ok=True)
        def copy(src, name=None):
            name = name or os.path.basename(src); dst = os.path.join(fdir, name)
            if src.lower().endswith(".md"):
                open(dst, "w", encoding="utf8").write(read_text(src))       # 统一转成 UTF-8
            else: shutil.copyfile(src, dst)
            return name, os.path.getsize(src)
        # 报告
        rep_html = []
        for src, nm, ds in st["reports"]:
            ext = src.rsplit(".", 1)[1].lower()
            fname = os.path.basename(src) if not src.endswith("index.html") else "板块顺势可视化报告.html"
            fn, sz = copy(src, fname); stats["r"] += 1
            rep_html.append(f'<a class="file" href="../{u("files", slug, fn)}" target="_blank" rel="noopener"><span class="ftype t-{ext}">{ICON[ext]}</span>'
                            f'<span class="fmain"><b>{E(nm)}</b><span>{E(ds)}</span></span><span class="fsize">{human(sz)}</span></a>')
        # 代码
        cdir = os.path.join(ROOT, "code", slug); os.makedirs(cdir, exist_ok=True)
        code_rows = []; code_items = []
        for idx, (src, ver, note) in enumerate(st["code"]):
            t = read_text(src); fn, sz = copy(src); stats["c"] += 1
            title_line = code_note(t); detail = note or code_detail(t)
            code_items.append((idx, ver, fn, sz, t, title_line, detail))
            code_rows.append(f'<tr><td><a href="../{u("code", slug, f"{idx+1}.html")}"><b>{E(ver)}</b></a></td><td class="cn">{E(detail or title_line)}</td>'
                             f'<td class="num">{t.count(chr(10)) + 1} 行</td><td class="act"><a href="../{u("code", slug, f"{idx+1}.html")}">查看</a>'
                             f'<a href="../{u("files", slug, fn)}" download>下载</a></td></tr>')
        for idx, ver, fn, sz, t, title_line, detail in code_items:
            others = "".join(f'<a class="vchip{" on" if j == idx else ""}" href="{j+1}.html">{E(v)}</a>' for j, v, *_ in code_items)
            body = f"""<nav class="crumb"><a class="back" href="../../s/{slug}.html" data-back>返回</a></nav>
<section class="codehead"><h1>{E(st['title'])} · {E(ver)}</h1><p class="sub">{E(title_line)}</p>{f'<p class="sub2">{E(detail)}</p>' if detail and detail != title_line else ''}
<div class="vers">{others}</div>
<div class="tools"><span class="mono">{E(fn)} · {t.count(chr(10)) + 1} 行 · {human(sz)}</span><span class="grow"></span>
<button class="btn" id="copy" type="button">复制代码</button><a class="btn" href="../../{u('files', slug, fn)}" download>下载 .py</a></div></section>
<div class="codebox">{highlight(t, PythonLexer(), fmt)}</div>
<textarea id="raw" hidden>{E(t)}</textarea>
<script>
document.getElementById("copy").addEventListener("click", async e => {{ const b = e.currentTarget, t = document.getElementById("raw").value;
  try {{ await navigator.clipboard.writeText(t); }} catch (err) {{ const a = document.getElementById("raw"); a.hidden = false; a.select(); document.execCommand("copy"); a.hidden = true; }}
  b.textContent = "已复制"; setTimeout(() => b.textContent = "复制代码", 1500); }});
</script>"""
            open(os.path.join(cdir, f"{idx+1}.html"), "w", encoding="utf8").write(page(f"{st['title']} {ver} 代码", body, 2, title_line))
        # 文档
        ddir = os.path.join(ROOT, "docs", slug); os.makedirs(ddir, exist_ok=True); doc_html = []
        for k, (src, nm) in enumerate(st["docs"]):
            ext = src.rsplit(".", 1)[1].lower(); fn, sz = copy(src); stats["d"] += 1
            inner = md2html(read_text(src)) if ext == "md" else docx2html(src)
            body = f"""<nav class="crumb"><a class="back" href="../../s/{slug}.html" data-back>返回</a></nav>
<div class="tools"><span class="mono">{E(fn)} · {human(sz)}</span><span class="grow"></span><a class="btn" href="../../{u('files', slug, fn)}" download>下载原文件</a></div>
<article class="doc">{inner}</article>"""
            open(os.path.join(ddir, f"{k+1}.html"), "w", encoding="utf8").write(page(f"{nm} · {st['title']}", body, 2, nm))
            doc_html.append(f'<a class="file" href="../{u("docs", slug, f"{k+1}.html")}"><span class="ftype t-{ext}">{ICON[ext]}</span>'
                            f'<span class="fmain"><b>{E(nm)}</b><span>在线阅读 · 原文件 {ext.upper()}</span></span><span class="fsize">{human(sz)}</span></a>')
        # 数据
        data_html = []
        for src, nm in st["data"]:
            fn, sz = copy(src)
            data_html.append(f'<a class="file" href="../{u("files", slug, fn)}" download><span class="ftype t-xlsx">Excel</span>'
                             f'<span class="fmain"><b>{E(nm)}</b><span>下载</span></span><span class="fsize">{human(sz)}</span></a>')
        links = "".join(f'<a class="file" href="{E(url)}" target="_blank" rel="noopener"><span class="ftype t-html">链接</span><span class="fmain"><b>{E(nm)}</b><span>{E(url)}</span></span></a>' for url, nm in st["links"])
        catname = dict((k, n) for k, n, _ in CATS)[st["cat"]]
        kp = "".join(f'<div class="kpi"><div class="k">{E(a)}</div><div class="v">{E(b)}</div><div class="s">{E(c)}</div></div>' for a, b, c in st["metrics"])
        sec = lambda title, inner: f'<section class="blk"><h2>{title}</h2>{inner}</section>' if inner else ""
        body = f"""<nav class="crumb"><a class="back" href="../index.html#{st['cat']}" data-back>返回</a></nav>
<header class="shead"><div class="chips"><span class="chip c-{st['cat']}">{catname}</span><span class="chip">{E(st['status'])}</span></div>
<h1>{E(st['title'])}</h1><p class="lede">{E(st['tagline'])}</p></header>
<div class="kpis">{kp}</div>
<div class="cols">
{sec("规则要点", "<ul class='bul'>" + "".join(f"<li>{E(x)}</li>" for x in st["rules"]) + "</ul>")}
{sec("结论与注意", "<ul class='bul'>" + "".join(f"<li>{E(x)}</li>" for x in st["notes"]) + "</ul>")}
</div>
{sec(f"报告 <small>{len(rep_html)}</small>", '<div class="files">' + "".join(rep_html) + links + "</div>" if rep_html or links else "")}
{sec(f"策略代码 <small>{len(code_rows)} 个文件 · 聚宽 JoinQuant</small>", '<div class="tblwrap"><table class="ctbl"><thead><tr><th>版本</th><th>说明</th><th>长度</th><th></th></tr></thead><tbody>' + "".join(code_rows) + "</tbody></table></div>" if code_rows else "")}
{sec(f"说明文档 <small>{len(doc_html)}</small>", '<div class="files">' + "".join(doc_html) + "</div>" if doc_html else "")}
{sec(f"数据 <small>{len(data_html)}</small>", '<div class="files">' + "".join(data_html) + "</div>" if data_html else "")}"""
        os.makedirs(os.path.join(ROOT, "s"), exist_ok=True)
        open(os.path.join(ROOT, "s", f"{slug}.html"), "w", encoding="utf8").write(page(st["title"] + " · 量化策略库", body, 1, st["tagline"]))
        cards[st["cat"]].append(f"""<a class="card" href="s/{slug}.html"><div class="chips"><span class="chip">{E(st['status'])}</span></div>
<h3>{E(st['title'])}</h3><p>{E(st['tagline'])}</p>
<div class="cm"><span class="cmk">2023 年至今收益率</span><span class="cmv">{E(st['ret'] or '无')}</span></div>
<div class="cnt">{"报告 %d · " % len(st['reports']) if st['reports'] else ""}代码 {len(st['code'])}{" · 文档 %d" % len(st['docs']) if st['docs'] else ""}{" · 数据 %d" % len(st['data']) if st['data'] else ""}</div></a>""")
    secs = ""
    for k, n, d in CATS:
        secs += f"""<section class="cat" id="{k}"><div class="cathead"><h2><span class="dot c-{k}"></span>{n} <small>{len(cards[k])}</small></h2><p>{E(d)}</p></div>
<div class="cards">{''.join(cards[k]) or '<p class="empty">暂无</p>'}</div></section>"""
    body = f"""<header class="hero"><h1>量化策略库</h1>
<p class="lede">我做过的 A 股策略：每个策略有规则说明、回测报告和可以直接复制到聚宽（JoinQuant）运行的代码。</p>
<div class="stats"><span><b>{stats['s']}</b> 个策略</span><span><b>{stats['r']}</b> 份报告</span><span><b>{stats['c']}</b> 个代码文件</span><span><b>{stats['d']}</b> 份说明文档</span></div>
<nav class="catnav">{"".join(f'<a href="#{k}"><span class="dot c-{k}"></span>{n}<small>{len(cards[k])}</small></a>' for k, n, _ in CATS)}</nav></header>
{secs}"""
    open(os.path.join(ROOT, "index.html"), "w", encoding="utf8").write(page("量化策略库", body, 0, "A 股量化策略：股息率、趋势、消息股策略的报告与聚宽代码"))
    open(os.path.join(ROOT, ".nojekyll"), "w").write("")
    print("done", stats)

if __name__ == "__main__":
    build()
