import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// Direct ANSI code to CSS color mapping (for inline styles)
// Note: Some models incorrectly use 37/38/39 as colors when they have special meanings
// We provide fallback colors to improve UX even with incorrect model outputs
const ANSI_FG_COLORS: Record<number, string> = {
  30: '#64748b', // Black → slate-500 (visible on dark bg)
  31: '#f87171', 32: '#4ade80', 33: '#facc15',
  34: '#60a5fa', 35: '#c084fc', 36: '#22d3ee',
  37: '#ffffff', // White → pure white for visibility
  38: '#fb923c', // Extended (fallback: orange for models that misuse it)
  39: '#86efac', // Default (fallback: light green for models that misuse it)
  90: '#94a3b8', 91: '#fca5a5', 92: '#86efac', 93: '#fde047',
  94: '#93c5fd', 95: '#d8b4fe', 96: '#67e8f9', 97: '#ffffff',
};

const ANSI_BG_COLORS: Record<number, string> = {
  40: '#1e293b', 41: 'rgba(239,68,68,0.2)', 42: 'rgba(34,197,94,0.2)', 43: 'rgba(234,179,8,0.2)',
  44: 'rgba(59,130,246,0.2)', 45: 'rgba(168,85,247,0.2)', 46: 'rgba(6,182,212,0.2)', 47: 'rgba(100,116,139,0.2)',
  100: 'rgba(148,163,184,0.2)', 101: 'rgba(252,165,165,0.2)', 102: 'rgba(134,239,172,0.2)', 103: 'rgba(253,224,71,0.2)',
  104: 'rgba(147,197,253,0.2)', 105: 'rgba(216,180,254,0.2)', 106: 'rgba(103,232,249,0.2)', 107: 'rgba(241,245,249,0.2)',
};

// Generate dynamic color from ANSI 256-color palette
const ansi256ToHex = (n: number): string => {
  if (n < 16) {
    // Standard colors (0-15)
    const std = [
      '#000000', '#aa0000', '#00aa00', '#aa5500', '#0000aa', '#aa00aa', '#00aaaa', '#aaaaaa',
      '#555555', '#ff5555', '#55ff55', '#ffff55', '#5555ff', '#ff55ff', '#55ffff', '#ffffff'
    ];
    return std[n] || '#ffffff';
  } else if (n < 232) {
    // 216 color cube (16-231)
    const i = n - 16;
    const r = Math.floor(i / 36) * 51;
    const g = Math.floor((i % 36) / 6) * 51;
    const b = (i % 6) * 51;
    return `rgb(${r},${g},${b})`;
  } else {
    // Grayscale (232-255)
    const gray = (n - 232) * 10 + 8;
    return `rgb(${gray},${gray},${gray})`;
  }
};

interface AnsiStyle {
  color?: string;
  backgroundColor?: string;
  fontWeight?: string;
  fontStyle?: string;
  textDecoration?: string;
  padding?: string;
  borderRadius?: string;
}

/**
 * Renders text with ANSI escape codes as colored React elements.
 * Supports standard colors (30-37, 90-97), backgrounds (40-47, 100-107),
 * 256-color palette (38;5;N), TrueColor (38;2;R;G;B), and text styles (bold, italic, underline).
 */
export const AnsiText = ({ text }: { text: string }) => {
  if (!text) return null;
  
  // Comprehensive regex to match ANSI escape codes in various forms
  // Handles both raw ESC byte (\x1b) and literal backslash sequences from JSON
  // Also accepts commas as delimiters (some models use \033[38;2;255,0,0m instead of \033[38;2;255;0;0m)
  const ansiRegex = /(?:\x1b\[|\x1B\[|\\033\[|\\x1b\[|\\e\[|\\u001b\[|\^\[\[)(\d+(?:[;,]\d+)*)m/g;
  
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let currentStyle: AnsiStyle = {};
  let match;
  
  while ((match = ansiRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      const segment = text.substring(lastIndex, match.index);
      const hasStyle = Object.keys(currentStyle).length > 0;
      if (hasStyle) {
        parts.push(
          <span key={lastIndex} style={{...currentStyle}}>
            {segment}
          </span>
        );
      } else {
        parts.push(<React.Fragment key={lastIndex}>{segment}</React.Fragment>);
      }
    }
    
    const codesStr = match[1] || "";
    const codes = codesStr.split(/[;,]/).map((c: string) => parseInt(c, 10));
    
    let i = 0;
    while (i < codes.length) {
      const code = codes[i];
      
      if (code === 0 || isNaN(code)) {
        currentStyle = {};
      } else if (code === 1) {
        currentStyle.fontWeight = 'bold';
      } else if (code === 3) {
        currentStyle.fontStyle = 'italic';
      } else if (code === 4) {
        currentStyle.textDecoration = 'underline';
      } else if (code >= 30 && code <= 37) {
        currentStyle.color = ANSI_FG_COLORS[code];
      } else if (code === 38) {
        // Extended foreground: 38;5;N or 38;2;R;G;B
        if (codes[i + 1] === 5 && codes[i + 2] !== undefined) {
          currentStyle.color = ansi256ToHex(codes[i + 2]);
          i += 2;
        } else if (codes[i + 1] === 2 && codes[i + 4] !== undefined) {
          currentStyle.color = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`;
          i += 4;
        } else {
          currentStyle.color = ANSI_FG_COLORS[38]; // Fallback: orange
        }
      } else if (code === 39) {
        currentStyle.color = ANSI_FG_COLORS[39]; // Fallback: light green
      } else if (code >= 40 && code <= 47) {
        currentStyle.backgroundColor = ANSI_BG_COLORS[code];
        currentStyle.padding = '0 0.25rem';
        currentStyle.borderRadius = '0.25rem';
      } else if (code === 48) {
        // Extended background: 48;5;N or 48;2;R;G;B
        if (codes[i + 1] === 5 && codes[i + 2] !== undefined) {
          currentStyle.backgroundColor = ansi256ToHex(codes[i + 2]);
          currentStyle.padding = '0 0.25rem';
          currentStyle.borderRadius = '0.25rem';
          i += 2;
        } else if (codes[i + 1] === 2 && codes[i + 4] !== undefined) {
          currentStyle.backgroundColor = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`;
          currentStyle.padding = '0 0.25rem';
          currentStyle.borderRadius = '0.25rem';
          i += 4;
        }
      } else if (code === 49) {
        delete currentStyle.backgroundColor;
        delete currentStyle.padding;
        delete currentStyle.borderRadius;
      } else if (code >= 90 && code <= 97) {
        currentStyle.color = ANSI_FG_COLORS[code];
      } else if (code >= 100 && code <= 107) {
        currentStyle.backgroundColor = ANSI_BG_COLORS[code];
        currentStyle.padding = '0 0.25rem';
        currentStyle.borderRadius = '0.25rem';
      }
      i++;
    }
    
    lastIndex = ansiRegex.lastIndex;
  }
  
  if (lastIndex < text.length) {
    const segment = text.substring(lastIndex);
    const hasStyle = Object.keys(currentStyle).length > 0;
    if (hasStyle) {
      parts.push(
        <span key={lastIndex} style={{...currentStyle}}>
          {segment}
        </span>
      );
    } else {
      parts.push(<React.Fragment key={lastIndex}>{segment}</React.Fragment>);
    }
  }
  
  return <>{parts.length > 0 ? parts : text}</>;
};

/**
 * Recursively processes React children to apply AnsiText to string nodes.
 */
export const RecursiveAnsi = (children: React.ReactNode): React.ReactNode => {
  return React.Children.map(children, child => {
    if (typeof child === 'string') return <AnsiText text={child} />;
    if (React.isValidElement(child) && (child.props as any).children) {
      return React.cloneElement(child as React.ReactElement<any>, {
        children: RecursiveAnsi((child.props as any).children)
      });
    }
    return child;
  });
};

/**
 * Renders Markdown with ANSI escape code support in text nodes.
 * Includes rehype-raw to allow rendering of raw HTML tags.
 */
export const MarkdownWithAnsi = ({ children }: { children: string }) => (
  <ReactMarkdown 
    remarkPlugins={[remarkGfm]}
    rehypePlugins={[rehypeRaw]}
    components={{
      code: ({ node, ...props }) => {
        if (typeof props.children === 'string') {
          return <code {...props}><AnsiText text={props.children} /></code>;
        }
        return <code {...props} />;
      },
      p: ({ children }) => <p>{RecursiveAnsi(children)}</p>,
      td: ({ children }) => <td>{RecursiveAnsi(children)}</td>,
      th: ({ children }) => <th>{RecursiveAnsi(children)}</th>,
      li: ({ children }) => {
        // Filter out whitespace-only text nodes that react-markdown might inject between blocks
        const cleanChildren = React.Children.toArray(children).filter(child => 
          typeof child !== 'string' || child.trim() !== ''
        );
        return <li>{RecursiveAnsi(cleanChildren)}</li>;
      },
      // @ts-ignore - Handle span if present
      span: ({ children }) => <span>{RecursiveAnsi(children)}</span>,
    }}
  >
    {children}
  </ReactMarkdown>
);

/**
 * Renders Markdown with ANSI support but WITHOUT interpreting raw HTML.
 * Use this for reasoning/thinking content where the model may discuss HTML tags
 * that should be displayed literally (e.g., "<table>", "<td>") instead of being
 * interpreted as actual HTML elements.
 */
export const MarkdownWithAnsiNoHtml = ({ children }: { children: string }) => (
  <ReactMarkdown 
    remarkPlugins={[remarkGfm]}
    // Note: No rehype-raw here - HTML tags will be displayed as text
    components={{
      code: ({ node, ...props }) => {
        if (typeof props.children === 'string') {
          return <code {...props}><AnsiText text={props.children} /></code>;
        }
        return <code {...props} />;
      },
      p: ({ children }) => <p>{RecursiveAnsi(children)}</p>,
      td: ({ children }) => <td>{RecursiveAnsi(children)}</td>,
      th: ({ children }) => <th>{RecursiveAnsi(children)}</th>,
      li: ({ children }) => {
        const cleanChildren = React.Children.toArray(children).filter(child => 
          typeof child !== 'string' || child.trim() !== ''
        );
        return <li>{RecursiveAnsi(cleanChildren)}</li>;
      },
      // @ts-ignore - Handle span if present
      span: ({ children }) => <span>{RecursiveAnsi(children)}</span>,
    }}
  >
    {children}
  </ReactMarkdown>
);

export default MarkdownWithAnsi;
