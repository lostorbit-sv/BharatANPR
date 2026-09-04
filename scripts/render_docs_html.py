import os, json
from pathlib import Path

base_dir = Path(r"C:\Users\amita\OneDrive\Desktop\anpr v1.0.2")
docs_dir = base_dir / "docs"

html_head = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            box-sizing: border-box;
            min-width: 200px;
            max-width: 1040px;
            margin: 0 auto;
            padding: 40px 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
            background-color: #f6f8fa;
        }
        .markdown-body {
            background-color: #ffffff;
            padding: 45px 55px;
            border-radius: 10px;
            border: 1px solid #d0d7de;
            box-shadow: 0 3px 12px rgba(140, 149, 159, 0.15);
        }
        .nav-bar {
            margin-bottom: 20px;
            padding: 12px 20px;
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            display: flex;
            gap: 15px;
            font-weight: 600;
        }
        .nav-bar a {
            color: #0969da;
            text-decoration: none;
        }
        .nav-bar a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="nav-bar">
        <span>Quick Navigation:</span>
        <a href="README.html">Project README</a>
        <a href="INTERVIEW_AND_ENGINEERING_GUIDE.html">Interview & Engineering Guide</a>
        <a href="MODEL_MATHEMATICS_AND_METHODS.html">Mathematics & Formulas Guide</a>
    </div>
    <div id="content" class="markdown-body"></div>
    <script>
        const raw = {raw_json};
        document.getElementById('content').innerHTML = marked.parse(raw);
    </script>
</body>
</html>
"""

files_to_convert = [
    (base_dir / "README.md", docs_dir / "README.html", "BharatANPR - Project Overview"),
    (docs_dir / "INTERVIEW_AND_ENGINEERING_GUIDE.md", docs_dir / "INTERVIEW_AND_ENGINEERING_GUIDE.html", "ANPR Interview & Engineering Guide"),
    (docs_dir / "MODEL_MATHEMATICS_AND_METHODS.md", docs_dir / "MODEL_MATHEMATICS_AND_METHODS.html", "Deep Learning Models & Mathematical Formulations")
]

for md_path, out_html, title in files_to_convert:
    if md_path.exists():
        raw_text = md_path.read_text(encoding="utf-8")
        json_str = json.dumps(raw_text)
        rendered = html_head.replace("{title}", title).replace("{raw_json}", json_str)
        out_html.write_text(rendered, encoding="utf-8")
        print(f"Rendered: {out_html.name}")

print("All documentation HTML files generated successfully in docs/!")
