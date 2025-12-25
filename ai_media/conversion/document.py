"""
Document conversion module for AI-Media.

Supports: md, html, pdf, docx, rtf, txt, json conversions.
Uses Markdown as intermediate format for hub-and-spoke conversion.
"""

import os
import json as json_module
from pathlib import Path

from ..utils.interaction import check_overwrite

SUPPORTED_FORMATS = ["md", "html", "pdf", "docx", "rtf", "txt", "json", "xhtml"]


def _read_to_markdown(input_path, input_format):
    """Read input file and convert to Markdown."""
    markdown_content = ""
    
    if input_format == "md":
        with open(input_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
    
    elif input_format in ["html", "xhtml"]:
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            with open(input_path, "r", encoding="utf-8") as f:
                markdown_content = h.handle(f.read())
        except ImportError:
            from bs4 import BeautifulSoup
            with open(input_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                markdown_content = soup.get_text()
            print("   ⚠️ html2text not installed, using basic text extraction")
    
    elif input_format == "docx":
        import docx
        doc = docx.Document(input_path)
        lines = []
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading 1'):
                lines.append(f"# {para.text}")
            elif para.style.name.startswith('Heading 2'):
                lines.append(f"## {para.text}")
            elif para.style.name.startswith('Heading 3'):
                lines.append(f"### {para.text}")
            else:
                lines.append(para.text)
        markdown_content = "\n\n".join(lines)
    
    elif input_format == "txt":
        with open(input_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
    
    elif input_format == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json_module.load(f)
        if isinstance(data, dict):
            markdown_content = data.get("content", "") or data.get("markdown", "") or data.get("text", "") or str(data)
        else:
            markdown_content = str(data)
    
    elif input_format == "pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(input_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            markdown_content = "\n\n".join(text_parts)
            print("   ⚠️ PDF conversion extracts text only (formatting/images lost)")
        except ImportError:
            raise ImportError("PyPDF2 required for PDF reading. Install: pip install PyPDF2")
    
    elif input_format == "rtf":
        try:
            from striprtf.striprtf import rtf_to_text
            with open(input_path, "r", encoding="utf-8") as f:
                markdown_content = rtf_to_text(f.read())
            print("   ⚠️ RTF conversion extracts text only (formatting lost)")
        except ImportError:
            raise ImportError("striprtf required for RTF reading. Install: pip install striprtf")
    
    return markdown_content


def _write_from_markdown(markdown_content, output_path, output_format):
    """Write Markdown content to output format."""
    import markdown as md_module
    
    if output_format == "md":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
    elif output_format in ["html", "xhtml"]:
        html = md_module.markdown(markdown_content, extensions=['extra', 'codehilite'])
        full_html = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Document</title>"
            f"<style>body{{font-family:sans-serif;max-width:800px;margin:2em auto;padding:1em;line-height:1.6}}"
            f"pre{{background:#f4f4f4;padding:1em;border-radius:5px}}</style></head>"
            f"<body>{html}</body></html>"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)
            
    elif output_format == "json":
        data = {"content": markdown_content, "html": md_module.markdown(markdown_content)}
        with open(output_path, "w", encoding="utf-8") as f:
            json_module.dump(data, f, indent=2)

    elif output_format == "txt":
        from bs4 import BeautifulSoup
        html = md_module.markdown(markdown_content)
        text = BeautifulSoup(html, "html.parser").get_text()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            
    elif output_format == "docx":
        import docx
        doc = docx.Document()
        for line in markdown_content.split('\n'):
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                doc.add_heading(line[4:], level=3)
            else:
                doc.add_paragraph(line)
        doc.save(output_path)
        
    elif output_format == "pdf":
        from xhtml2pdf import pisa
        html_content = md_module.markdown(markdown_content, extensions=['extra', 'fenced_code', 'tables'])
        full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body {{ font-family: Helvetica, sans-serif; font-size: 10pt; }}</style>
</head><body>{html_content}</body></html>"""
        with open(output_path, "wb") as f:
            pisa.CreatePDF(full_html, dest=f)
            
    elif output_format == "rtf":
        # Basic RTF generation
        rtf_lines = [r'{\rtf1\ansi\deff0', r'{\fonttbl{\f0 Helvetica;}}', r'\f0\fs24']
        for line in markdown_content.split('\n'):
            escaped = line.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
            if line.startswith('# '):
                rtf_lines.append(r'\pard\sb400\sa200\b\fs48 ' + escaped[2:] + r'\b0\fs24\par')
            elif line.startswith('## '):
                rtf_lines.append(r'\pard\sb300\sa150\b\fs36 ' + escaped[3:] + r'\b0\fs24\par')
            else:
                rtf_lines.append(r'\pard\sa100 ' + escaped + r'\par')
        rtf_lines.append('}')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(rtf_lines))


def convert_document(input_path, target):
    """Convert document format using MD as intermediate hub.
    
    Args:
        input_path: Source document file
        target: Output path or format
        
    Supported formats: md, html, pdf, docx, rtf, txt, json
    """
    # Determine output path and format
    target = target.strip().lower()
    if '/' in target or '\\' in target:
        output_path = target
        output_format = Path(target).suffix.lstrip('.').lower()
    elif target.startswith('.'):
        output_path = f"{Path(input_path).stem}{target}"
        output_format = target.lstrip('.').lower()
    else:
        output_path = f"{Path(input_path).stem}.{target}"
        output_format = target.lower()
    
    if output_format not in SUPPORTED_FORMATS:
        print(f"❌ Unsupported output format: {output_format}")
        print(f"   Supported: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    # Determine input format
    input_format = Path(input_path).suffix.lstrip('.').lower()
    if input_format not in SUPPORTED_FORMATS:
        print(f"❌ Unsupported input format: {input_format}")
        print(f"   Supported: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    print(f"📄 Converting Document: {input_path}")
    print(f"   {input_format.upper()} → {output_format.upper()}")
    
    should_write, output_path, _, _ = check_overwrite(output_path, always_overwrite=os.environ.get("AI_MEDIA_FORCE") == "1")
    if not should_write:
        return False
    
    try:
        # Step 1: Read to Markdown
        markdown_content = _read_to_markdown(input_path, input_format)
        
        if not markdown_content.strip():
            print("❌ No content extracted from input file")
            return False
        
        # Step 2: Write to target format
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _write_from_markdown(markdown_content, output_path, output_format)
        
        print(f"✅ Saved to {output_path}")
        return True
        
    except ImportError as e:
        print(f"❌ {e}")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
