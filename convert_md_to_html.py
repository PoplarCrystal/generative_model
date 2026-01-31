#!/usr/bin/env python3
"""
将AE系列篇的markdown文件转换为HTML格式
支持数学公式渲染（LaTeX）
"""

import os
import re
from pathlib import Path

try:
    import markdown
    from markdown.extensions import codehilite, fenced_code, tables
    from pymdownx import arithmetic, superfences, emoji
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    print("警告: markdown库未安装，尝试使用pandoc...")

def convert_md_to_html_markdown(md_content, title):
    """使用Python markdown库转换"""
    # 配置markdown扩展
    extensions = [
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'pymdownx.arithmatex',
        'pymdownx.superfences',
        'pymdownx.emoji',
    ]
    
    extension_configs = {
        'pymdownx.arithmatex': {
            'generic': True,
        },
        'codehilite': {
            'css_class': 'highlight',
            'use_pygments': True,
        }
    }
    
    # 转换为HTML
    md = markdown.Markdown(extensions=extensions, extension_configs=extension_configs)
    html_body = md.convert(md_content)
    
    # 添加MathJax支持数学公式
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background-color: #fff;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }}
        .MathJax {{
            font-size: 1.1em;
        }}
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }}
        }};
    </script>
</head>
<body>
{html_body}
</body>
</html>"""
    
    return html_template

def convert_md_to_html_pandoc(md_file, html_file, title):
    """使用pandoc转换（如果可用），并后处理HTML以修复MathJax和样式"""
    import subprocess
    try:
        # 使用pandoc转换，指定MathJax CDN
        subprocess.run([
            'pandoc',
            md_file,
            '-o', html_file,
            '--mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
            '--standalone',
        ], check=True)
        
        # 读取生成的HTML并修复
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 替换或添加MathJax脚本和配置
        mathjax_script = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>'
        mathjax_config = '''<script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };
        </script>'''
        
        # 移除所有现有的MathJax相关script标签
        html_content = re.sub(
            r'<script[^>]*mathjax[^>]*>.*?</script>',
            '',
            html_content,
            flags=re.DOTALL | re.IGNORECASE
        )
        html_content = re.sub(
            r'<script[^>]*src=["\'][^"\']*mathjax[^"\']*["\'][^>]*></script>',
            '',
            html_content,
            flags=re.IGNORECASE
        )
        
        # 在</head>之前添加MathJax脚本和配置
        html_content = html_content.replace('</head>', 
            mathjax_script + '\n' + mathjax_config + '\n</head>')
        
        # 添加更好的CSS样式
        enhanced_css = '''
        <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            line-height: 1.7;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px 40px;
            color: #333;
            background-color: #fff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            font-weight: 600;
        }
        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2em;
        }
        h2 {
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            font-size: 1.6em;
            margin-top: 2em;
        }
        h3 {
            font-size: 1.3em;
            margin-top: 1.5em;
        }
        h4 {
            font-size: 1.1em;
            margin-top: 1.2em;
        }
        p {
            margin: 1em 0;
            text-align: justify;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            line-height: 1.45;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
            font-size: 0.9em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1.5em 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 1.5em 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            color: #555;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5em auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 4px;
        }
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        li {
            margin: 0.5em 0;
        }
        .math {
            font-size: 1.05em;
        }
        .math.display {
            margin: 1.5em 0;
            text-align: center;
        }
        .math.inline {
            margin: 0 2px;
        }
        </style>'''
        
        # 在</head>之前添加CSS（如果还没有的话）
        if '</style>' not in html_content or 'font-family' not in html_content:
            html_content = html_content.replace('</head>', enhanced_css + '\n</head>')
        
        # 更新title
        if title:
            html_content = re.sub(
                r'<title>.*?</title>',
                f'<title>{title}</title>',
                html_content,
                flags=re.DOTALL
            )
        
        # 写回文件
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Pandoc错误: {e}")
        return False

def main():
    # 获取所有AE系列篇的markdown文件
    base_dir = Path(__file__).parent
    md_files = sorted(base_dir.glob('AE系列篇-*.md'))
    
    if not md_files:
        print("未找到AE系列篇的markdown文件")
        return
    
    print(f"找到 {len(md_files)} 个markdown文件")
    
    # 转换每个文件
    for md_file in md_files:
        print(f"正在转换: {md_file.name}")
        
        # 读取markdown内容
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 提取标题（从第一行或文件名）
        title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        else:
            title = md_file.stem
        
        # 生成HTML文件名
        html_file = md_file.with_suffix('.html')
        
        # 转换 - 优先使用pandoc（对数学公式支持更好）
        import subprocess
        try:
            subprocess.run(['pandoc', '--version'], check=True, capture_output=True)
            # 使用pandoc转换
            if convert_md_to_html_pandoc(str(md_file), str(html_file), title):
                print(f"  ✓ 已生成: {html_file.name} (使用pandoc)")
            else:
                print(f"  ✗ 转换失败")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果没有pandoc，尝试使用markdown库
            if HAS_MARKDOWN:
                html_content = convert_md_to_html_markdown(md_content, title)
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"  ✓ 已生成: {html_file.name} (使用markdown库)")
            else:
                print(f"  ✗ 转换失败: 请安装pandoc或markdown库")
    
    print("\n转换完成！")

if __name__ == '__main__':
    main()
