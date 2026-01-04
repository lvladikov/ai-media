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
            from pypdf import PdfReader
            reader = PdfReader(input_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            markdown_content = "\n\n".join(text_parts)
            
            if not markdown_content.strip():
                raise ValueError("No text found in PDF. The file might be a scanned image (OCR not supported).")
                
            print("   ⚠️ PDF conversion extracts text only (formatting/images lost)")
        except ImportError:
            raise ImportError("pypdf required for PDF reading. Install: pip install pypdf")
    
    elif input_format == "rtf":
        try:
            from striprtf.striprtf import rtf_to_text
            with open(input_path, "r", encoding="utf-8") as f:
                markdown_content = rtf_to_text(f.read())
            print("   ⚠️ RTF conversion extracts text only (formatting lost)")
        except ImportError:
            # Fallback for when striprtf is not available
            import re
            def fallback_rtf_to_text(rtf_str):
                # Very basic RTF stripper
                pattern = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I)
                destinations = frozenset((
                    'ftnid','ftnsep','ftnsepc','annotation','atnid','atnref','atntime','atrfend','atrfstart',
                    'author','background','bkmkend','bkmkstart','blipuid','buptim','category','colorschememapping',
                    'colortbl','comment','company','creatim','datafield','datastore','defchp','defpap','do',
                    'doccomm','docvar','dptxbxtext','ebcend','ebcstart','factoidname','falt','fchars','ffdeftext',
                    'ffentrymcr','ffexitmcr','ffformat','ffhelptext','ffl','ffname','ffstattext','field',
                    'filetbl','fldinst','fldrslt','fldtype','fname','fontemb','fontfile','fonttbl','footer',
                    'footerf','footerl','footerr','footnote','formfield','generator','gridtbl','header',
                    'headerf','headerl','headerr','hl','hlfr','hllink','hlsrc','hsv','htmltag','info',
                    'keycode','keywords','latency','lchars','levelnumbers','leveltext','list','listlevel',
                    'listname','listoverride','listoverridetable','listpicture','liststylename','listtable',
                    'listtext','lsdlockedexcept','macc','maccPr','mailmerge','malformed','margins','mati',
                    'matn','mhtmltag','mmath','mmathPr','mnum','mpocket','mtype','mxml','nesttableprops',
                    'nextfile','nonesttables','obj','objalias','objclass','objdata','object','objname',
                    'objsect','objtime','oldcprops','oldpprops','oldsprops','oldtprops','oleclsid','operator',
                    'panose','password','passwordhash','pgp','pgptbl','picprop','pict','pn','pnseclvl',
                    'pntext','pntxta','pntxtb','printim','private','propname','protend','protstart','protusertbl',
                    'pxe','result','revtbl','revtim','rsidtbl','rxe','shp','shpgrp','shpinst','shppict',
                    'shprslt','shptxt','sn','sp','staticval','stylesheet','subject','sv','svb','tc',
                    'template','themedata','title','txe','ud','upr','userprops','wgrffmtfilter','windowcaption',
                    'writereservation','writereservhash','xe','xform','xmlattrname','xmlattrvalue','xmlclose',
                    'xmlname','xmlnstbl','xmlopen',
                ))
                stack = []
                ignorable = False       # Whether this group (and all inside it) are "ignorable".
                ucskip = 1              # Number of ASCII characters to skip after a unicode character.
                curskip = 0             # Number of ASCII characters left to skip
                out = []                # Output buffer.

                for match in pattern.finditer(rtf_str):
                    word,arg,hex,char,brace,tchar = match.groups()
                    if brace:
                        curskip = 0
                        if brace == '{':
                            stack.append((ucskip,ignorable))
                        elif brace == '}':
                            if stack:
                                ucskip,ignorable = stack.pop()
                    elif char: # \x (not a letter)
                        curskip = 0
                        if char == '~':
                            if not ignorable: out.append('\xA0')
                        elif char in '{}\\':
                            if not ignorable: out.append(char)
                        elif char == '*':
                            ignorable = True
                    elif word: # \foo
                        curskip = 0
                        if word in destinations:
                            ignorable = True
                        elif ignorable:
                            pass
                        elif word == 'par' or word == 'line' or word == 'row':
                            out.append('\n')
                        elif word == 'tab':
                            out.append('\t')
                    elif hex: # \'xx
                        if curskip > 0:
                            curskip -= 1
                        elif not ignorable:
                            out.append(chr(int(hex,16)))
                    elif tchar:
                        if curskip > 0:
                            curskip -= 1
                        elif not ignorable:
                            out.append(tchar)
                return "".join(out)
            
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                markdown_content = fallback_rtf_to_text(f.read())
            print("   ⚠️ RTF conversion using fallback regex parser (formatting lost)")

    
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
        
        # Determine output format early for writing
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Write to target format
        _write_from_markdown(markdown_content, output_path, output_format)
        
        print(f"✅ Saved to {output_path}")
        return True
        
    except Exception as e:
        # Re-raise the exception so the task runner gets the specific error message
        # The runner will handle logging and status update
        raise e
