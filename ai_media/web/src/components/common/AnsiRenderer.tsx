import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// Direct ANSI code to CSS color mapping
const ANSI_FG_COLORS: Record<number, string> = {
  30: '#64748b', // Black → slate-500
  31: '#f87171', 32: '#4ade80', 33: '#facc15',
  34: '#60a5fa', 35: '#c084fc', 36: '#22d3ee',
  37: '#ffffff', // White
  38: '#fb923c', // Extended (fallback: orange)
  39: '#86efac', // Default (fallback: light green)
  90: '#94a3b8', 91: '#fca5a5', 92: '#86efac', 93: '#fde047',
  94: '#93c5fd', 95: '#d8b4fe', 96: '#67e8f9', 97: '#ffffff',
};

const ANSI_BG_COLORS: Record<number, string> = {
  40: '#1e293b', 41: 'rgba(239,68,68,0.2)', 42: 'rgba(34,197,94,0.2)', 43: 'rgba(234,179,8,0.2)',
  44: 'rgba(59,130,246,0.2)', 45: 'rgba(168,85,247,0.2)', 46: 'rgba(6,182,212,0.2)', 47: 'rgba(100,116,139,0.2)',
  100: 'rgba(148,163,184,0.2)', 101: 'rgba(252,165,165,0.2)', 102: 'rgba(134,239,172,0.2)', 103: 'rgba(253,224,71,0.2)',
  104: 'rgba(147,197,253,0.2)', 105: 'rgba(216,180,254,0.2)', 106: 'rgba(103,232,249,0.2)', 107: 'rgba(241,245,249,0.2)',
};

const ansi256ToHex = (n: number): string => {
  if (n < 16) {
    const std = [
      '#000000', '#aa0000', '#00aa00', '#aa5500', '#0000aa', '#aa00aa', '#00aaaa', '#aaaaaa',
      '#555555', '#ff5555', '#55ff55', '#ffff55', '#5555ff', '#ff55ff', '#55ffff', '#ffffff'
    ];
    return std[n] || '#ffffff';
  } else if (n < 232) {
    const i = n - 16;
    const r = Math.floor(i / 36) * 51;
    const g = Math.floor((i % 36) / 6) * 51;
    const b = (i % 6) * 51;
    return `rgb(${r},${g},${b})`;
  } else {
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

// Regex to match ANSI escape codes 
// Updated to support double backslashes (\\033) which often bypass markdown escaping
const ANSI_REGEX = /(?:\x1b\[|\x1B\[|\\033\[|\\\\033\[|\\x1b\[|\\\\x1b\[|\\e\[|\\u001b\[|\^\[\[)(\d+(?:[;,]\d+)*)m/g;

/**
 * Parses text containing ANSI codes and returns a list of React nodes with styles applied.
 * Returns the final style state after processing the text.
 */
export const processAnsi = (text: string, initialStyle: AnsiStyle = {}): { nodes: React.ReactNode[], finalStyle: AnsiStyle } => {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let currentStyle: AnsiStyle = { ...initialStyle };
  let match;

  // Reset regex state
  ANSI_REGEX.lastIndex = 0;

  while ((match = ANSI_REGEX.exec(text)) !== null) {
    if (match.index > lastIndex) {
      const segment = text.substring(lastIndex, match.index);
      if (Object.keys(currentStyle).length > 0) {
        parts.push(
          <span key={`${lastIndex}-${match.index}`} style={{ ...currentStyle }}>
            {segment}
          </span>
        );
      } else {
        parts.push(segment);
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
        if (codes[i + 1] === 5 && codes[i + 2] !== undefined) {
          currentStyle.color = ansi256ToHex(codes[i + 2]);
          i += 2;
        } else if (codes[i + 1] === 2 && codes[i + 4] !== undefined) {
          currentStyle.color = `rgb(${codes[i + 2]},${codes[i + 3]},${codes[i + 4]})`;
          i += 4;
        } else {
          currentStyle.color = ANSI_FG_COLORS[38];
        }
      } else if (code === 39) {
        currentStyle.color = ANSI_FG_COLORS[39];
      } else if (code >= 40 && code <= 47) {
        currentStyle.backgroundColor = ANSI_BG_COLORS[code];
        currentStyle.padding = '0 0.25rem';
        currentStyle.borderRadius = '0.25rem';
      } else if (code === 48) {
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

    lastIndex = ANSI_REGEX.lastIndex;
  }

  if (lastIndex < text.length) {
    const segment = text.substring(lastIndex);
    if (Object.keys(currentStyle).length > 0) {
      parts.push(
        <span key={`tail-${lastIndex}`} style={{ ...currentStyle }}>
          {segment}
        </span>
      );
    } else {
      parts.push(segment);
    }
  }

  return { nodes: parts, finalStyle: currentStyle };
};

/**
 * Standard component for rendering a single ANSI string (legacy usage)
 */
export const AnsiText = ({ text }: { text: string }) => {
  const { nodes } = processAnsi(text);
  return <>{nodes}</>;
};

/**
 * Recursively processes React children to apply AnsiText to string nodes,
 * persisting ANSI state across siblings.
 */
export const RecursiveAnsi = (children: React.ReactNode, initialStyle: AnsiStyle = {}): React.ReactNode => {
  const resultArray: React.ReactNode[] = [];
  let currentStyle = { ...initialStyle };

  React.Children.forEach(children, (child, index) => {
    if (typeof child === 'string') {
      const { nodes, finalStyle } = processAnsi(child, currentStyle);
      currentStyle = finalStyle;
      resultArray.push(...nodes);
    } 
    else if (React.isValidElement(child)) {
      // Check if the element is 'code' or 'pre' to prevent processing ANSI inside them
      const type = child.type;
      const isCodeOrPre = type === 'code' || type === 'pre';

      if (!isCodeOrPre && (child.props as any).children) {
        // Pass current style DOWN into the child element
        const processedChildren = RecursiveAnsi((child.props as any).children, currentStyle);
        
        // Merge current style into the element's style prop
        const existingStyle = (child.props as any).style || {};
        const mergedStyle = { ...currentStyle, ...existingStyle };

        resultArray.push(React.cloneElement(child as React.ReactElement<any>, {
          key: index,
          style: mergedStyle,
          children: processedChildren
        }));
      } else {
        // Element without children or excluded element (code/pre)
        // For code/pre, we do NOT merge the current ANSI style, keeping it clean
        const existingStyle = (child.props as any).style || {};
        const mergedStyle = isCodeOrPre ? existingStyle : { ...currentStyle, ...existingStyle };
        
        resultArray.push(React.cloneElement(child as React.ReactElement<any>, { 
          key: index, 
          style: mergedStyle 
        }));
      }
    } else {
      resultArray.push(child);
    }
  });

  return resultArray;
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
      p: ({ children }) => <p>{RecursiveAnsi(children)}</p>,
      td: ({ children }) => <td>{RecursiveAnsi(children)}</td>,
      th: ({ children }) => <th>{RecursiveAnsi(children)}</th>,
      li: ({ children }) => {
        const cleanChildren = React.Children.toArray(children).filter(child => 
          typeof child !== 'string' || child.trim() !== ''
        );
        return <li>{RecursiveAnsi(cleanChildren)}</li>;
      },
      // @ts-ignore
      span: ({ children }) => <span>{RecursiveAnsi(children)}</span>,
      // @ts-ignore
      strong: ({ children }) => <strong>{RecursiveAnsi(children)}</strong>,
      // @ts-ignore
      em: ({ children }) => <em>{RecursiveAnsi(children)}</em>,
    }}
  >
    {children}
  </ReactMarkdown>
);

/**
 * Renders Markdown with ANSI support but WITHOUT interpreting raw HTML.
 */
export const MarkdownWithAnsiNoHtml = ({ children }: { children: string }) => (
  <ReactMarkdown 
    remarkPlugins={[remarkGfm]}
    components={{
      p: ({ children }) => <p>{RecursiveAnsi(children)}</p>,
      td: ({ children }) => <td>{RecursiveAnsi(children)}</td>,
      th: ({ children }) => <th>{RecursiveAnsi(children)}</th>,
      li: ({ children }) => {
        const cleanChildren = React.Children.toArray(children).filter(child => 
          typeof child !== 'string' || child.trim() !== ''
        );
        return <li>{RecursiveAnsi(cleanChildren)}</li>;
      },
      // @ts-ignore
      span: ({ children }) => <span>{RecursiveAnsi(children)}</span>,
    }}
  >
    {children}
  </ReactMarkdown>
);

export default MarkdownWithAnsi;
