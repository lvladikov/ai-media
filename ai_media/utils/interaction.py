"""
Interactive Utilities for AI-Media.

Common CLI interaction functions for parsing input, displaying menus, and handling files.
Includes full mouse click support for interactive menus.
"""

import os
import sys
import re
import select

# =============================================================================
# Windows VT Mode (enables ANSI escape codes for colors/styling)
# Note: Mouse support is not available on Windows because msvcrt.getch()
# only captures keyboard input. Mouse events require Unix raw terminal mode.
# =============================================================================

def _enable_windows_vt_mode():
    """Enable Virtual Terminal (VT) mode on Windows for ANSI escape sequences.
    
    This enables colors, cursor positioning, and other ANSI output features.
    Mouse support NOT available - Windows msvcrt only reads keyboard events.
    """
    if os.name != 'nt':
        return
    
    try:
        import ctypes
        from ctypes import wintypes
        
        kernel32 = ctypes.windll.kernel32
        
        # Enable VT processing for output only
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        
        h_out = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = wintypes.DWORD()
        kernel32.GetConsoleMode(h_out, ctypes.byref(mode))
        kernel32.SetConsoleMode(h_out, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass  # Silently fail on older Windows or if not in a console

# Enable VT mode on import
_enable_windows_vt_mode()

# ANSI escape codes
CLEAR_LINE = "\033[K"
CYAN = "\033[96m"
RESET = "\033[0m"
DIM = "\033[90m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
UP = "\033[F"


import threading

_loading_timer = None
_loading_shown = False

def _show_loading_message():
    global _loading_shown
    _loading_shown = True
    print("⏳ Loading... (May take a moment while modules initialize and cache)", flush=True)


def emoji(emoji_char, fallback=""):
    """Return emoji if terminal supports it, otherwise return fallback text."""
    try:
        emoji_char.encode(sys.stdout.encoding or 'utf-8')
        return emoji_char
    except (UnicodeEncodeError, LookupError, AttributeError):
        return fallback


def clear_screen():
    """Clear terminal screen.
    
    Only clears if stdout is a TTY (interactive terminal).
    Skips clear when output is piped (e.g., during testing) to allow capture.
    """
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')


def show_header(title="AI-Media"):
    """Show interactive mode header."""
    print(f"\n{'═'*60}", flush=True)
    print(f"{emoji('🎨 ', '')}{title}", flush=True)
    print(f"{'═'*60}\n", flush=True)


# =============================================================================
# Terminal Raw Mode Helpers (for mouse support)
# =============================================================================

def get_cursor_position():
    """Query cursor position using ANSI DSR. Returns (row, col) or None."""
    if os.name == 'nt' or not sys.stdout.isatty():
        return None
    
    import termios, tty
    
    # Handle closed stdin (e.g., during automated testing on Mac/Linux)
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (OSError, IOError, ValueError, termios.error):
        return None
    
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[6n")
        sys.stdout.flush()
        
        # Wait for response
        if select.select([sys.stdin], [], [], 0.1)[0]:
            resp = ""
            while True:
                ch = sys.stdin.read(1)
                resp += ch
                if ch == 'R': break
            
            # Parse \033[<row>;<col>R
            match = re.search(r'\x1b\[(\d+);(\d+)R', resp)
            if match:
                return int(match.group(1)), int(match.group(2))
    except (termios.error, IOError, OSError, ValueError):
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (OSError, IOError, ValueError, termios.error):
            pass
    return None


def set_raw_mode(fd):
    """Enter raw mode, return old settings."""
    if os.name == 'nt':
        return None
    
    # Handle closed stdin (e.g., during automated testing on Mac/Linux)
    try:
        if not sys.stdin.isatty():
            return None
    except (OSError, IOError, ValueError):
        return None
    
    import termios, tty
    try:
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        return old
    except (termios.error, OSError, IOError, ValueError):
        return None


def restore_mode(fd, old_settings):
    """Restore terminal to old settings."""
    if os.name == 'nt' or not old_settings or fd is None:
        return
    import termios
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (termios.error, OSError, IOError, ValueError):
        pass  # Ignore errors when restoring (stdin might be closed)


# =============================================================================
# Key Input Handling (with mouse support)
# =============================================================================

def get_key():
    """Read a single key press from stdin (cross-platform).
    
    Returns:
        String for regular keys ('UP', 'DOWN', 'ENTER', etc.)
        Tuple ('MOUSE', x, y) for mouse clicks
        'SCROLL_UP' or 'SCROLL_DOWN' for scroll wheel
    """
    if os.name == 'nt':  # Windows (keyboard only, mouse not supported)
        import msvcrt
        ch = msvcrt.getch()
        
        # Handle extended keys (arrow keys, function keys etc.)
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            if ch2 == b'H': return 'UP'
            if ch2 == b'P': return 'DOWN'
            if ch2 == b'G': return 'HOME'
            if ch2 == b'O': return 'END'
            if ch2 == b'I': return 'PAGE_UP'
            if ch2 == b'Q': return 'PAGE_DOWN'
            return ch2.decode('utf-8', errors='ignore')
        
        if ch == b'\x1b': return 'ESC'
        if ch == b'\r': return 'ENTER'
        if ch == b'\x03': raise KeyboardInterrupt
        return ch.decode('utf-8', errors='ignore')
    else:  # Unix/Mac
        import tty, termios
        
        # Handle closed stdin (e.g., during automated testing on Mac/Linux)
        try:
            fd = sys.stdin.fileno()
        except (OSError, IOError, ValueError):
            # stdin is closed or not available - exit gracefully
            raise KeyboardInterrupt("stdin closed")
        
        is_interactive = sys.stdout.isatty()
        old_settings = None
        if is_interactive:
            old_settings = termios.tcgetattr(fd)
        
        try:
            if is_interactive:
                tty.setraw(fd)
            
            def read_bytes(n=1):
                return os.read(fd, n).decode('utf-8', errors='ignore')
            
            ch = read_bytes(1)
            
            if ch == '\x1b':  # Escape sequence
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    return 'ESC'
                
                ch2 = read_bytes(1)
                
                if ch2 == '[':
                    ch3 = read_bytes(1)
                    
                    # SGR Mouse: \033[<0;10;20M or \033[<0;10;20m
                    if ch3 == '<':
                        mouse_seq = ""
                        while True:
                            char = read_bytes(1)
                            if char in ('m', 'M'):
                                end_char = char
                                break
                            mouse_seq += char
                        
                        parts = mouse_seq.split(';')
                        if len(parts) >= 3:
                            btn = parts[0]
                            x = int(parts[1])
                            y = int(parts[2])
                            # Standard left click is 0, Right is 2
                            # Scroll Up is 64, Scroll Down is 65
                            if end_char == 'M':
                                if btn == '0': return ('MOUSE', x, y)
                                if btn == '64': return 'SCROLL_UP'
                                if btn == '65': return 'SCROLL_DOWN'
                            return None
                    
                    if ch3 == 'A': return 'UP'
                    if ch3 == 'B': return 'DOWN'
                    if ch3 == 'C': return 'RIGHT'
                    if ch3 == 'D': return 'LEFT'
                    if ch3 == 'H': return 'HOME'
                    if ch3 == 'F': return 'END'
                    if ch3 in ['1', '4', '5', '6']:
                        ch4 = read_bytes(1)
                        if ch4 == '~':
                            if ch3 == '1': return 'HOME'
                            if ch3 == '4': return 'END'
                            if ch3 == '5': return 'PAGE_UP'
                            if ch3 == '6': return 'PAGE_DOWN'
                            
        finally:
            if is_interactive and old_settings:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if ch == '\r' or ch == '\n': return 'ENTER'
        if ch == '\x03': raise KeyboardInterrupt
        return ch


# =============================================================================
# Interactive Menu with Mouse Support
# =============================================================================

def prompt_menu(prompt, options, allow_back=True, default_index=0, page_size=20):
    """Show interactive menu with arrow key navigation, pagination, and mouse support.
    
    Args:
        prompt: Header text to display (or None)
        options: List of (label, value) tuples
        allow_back: Whether to show a "Back" option
        default_index: Initially selected index
        page_size: Max items visible at once
        
    Returns:
        Selected value, or None if Back/cancelled
    """
    items = list(options)
    if allow_back:
        items.append((f"{emoji('⬅️  ', '')}Back", None))
    elif not options:
        return None

    current_idx = default_index if 0 <= default_index < len(items) else 0
    start_idx = 0

    # Hide cursor
    print(HIDE_CURSOR, end="")
    
    # --- MOUSE SUPPORT (Unix/Mac only) ---
    mouse_enabled = False
    menu_start_row = None
    
    # Enter persistent raw mode for mouse (Unix only)
    fd_raw = None
    old_raw = None
    if os.name != 'nt' and sys.stdout.isatty():
        fd_raw = sys.stdin.fileno()
        old_raw = set_raw_mode(fd_raw)
    
        # Enable Mouse Reporting (1000: basic, 1006: SGR extended)
        print("\033[?1000h\033[?1006h", end="", flush=True)
        # Query start row
        pos = get_cursor_position()
        if pos:
            menu_start_row = pos[0]  # ABS row (1-based)
            mouse_enabled = True
    # --- MOUSE SUPPORT END ---

    if prompt:
        # In raw mode, \n is just LF - need \r\n
        prompt_formatted = prompt.replace('\n', '\r\n')
        print(f"{prompt_formatted}\r")
        prompt_lines = prompt.count('\n') + 1
        if mouse_enabled:
            menu_start_row += prompt_lines

    max_view_lines = min(len(items), page_size) + 3
    for _ in range(max_view_lines):
        print("\r")
    
    print(UP * max_view_lines, end="", flush=True)
    
    # Re-query cursor position after reservation
    if mouse_enabled:
        pos = get_cursor_position()
        if pos:
            menu_start_row = pos[0]

    try:
        while True:
            # Pagination
            if current_idx < start_idx:
                start_idx = current_idx
            elif current_idx >= start_idx + page_size:
                start_idx = current_idx - page_size + 1
            
            end_idx = min(len(items), start_idx + page_size)
            visible_items = items[start_idx:end_idx]
            
            lines_printed = 0
            
            # Up indicator
            if start_idx > 0:
                print(f"{DIM}   ⬆️  ... ({start_idx} more above){RESET}{CLEAR_LINE}\r")
                lines_printed += 1
            
            # Menu items
            for i, (label, val) in enumerate(visible_items):
                abs_index = start_idx + i
                is_selected = (abs_index == current_idx)
                prefix = " > " if is_selected else "   "
                number = f"{abs_index+1}." if abs_index < len(options) else "0."
                
                if is_selected:
                    line = f"{CYAN}{prefix}{number:<4}  {label}{RESET}"
                else:
                    line = f"{prefix}{number:<4}  {label}"
                
                print(f"{line}{CLEAR_LINE}\r")
                lines_printed += 1
            
            # Down indicator
            if end_idx < len(items):
                remaining = len(items) - end_idx
                print(f"{DIM}   ⬇️  ... ({remaining} more below){RESET}{CLEAR_LINE}\r")
                lines_printed += 1
            
            # Clear extra lines
            for _ in range((max_view_lines - 1) - lines_printed):
                print(f"{CLEAR_LINE}\r")
            
            # Hint
            hint_back = ", '0' for Back" if allow_back else ""
            print(f"{DIM}(Tip: 'Home'/'End' or 'g'/'G' for top/bottom{hint_back}){RESET}{CLEAR_LINE}\r")
            
            # Move cursor back
            print(UP * max_view_lines, end="", flush=True)

            # Handle input
            key = get_key()
            
            # Skip None (e.g., unhandled mouse events)
            if key is None:
                continue
            
            # Scroll wheel
            if key == 'SCROLL_UP':
                current_idx = max(0, current_idx - 1)
                if current_idx < start_idx:
                    start_idx = current_idx
                continue
            elif key == 'SCROLL_DOWN':
                current_idx = min(len(items)-1, current_idx + 1)
                if current_idx >= start_idx + page_size:
                    start_idx = current_idx - page_size + 1
                continue
            
            # Mouse click
            if isinstance(key, tuple) and key[0] == 'MOUSE':
                mx, my = key[1], key[2]
                if menu_start_row:
                    rel_y = my - menu_start_row
                    
                    # Account for Up indicator
                    header_offset = 1 if start_idx > 0 else 0
                    
                    # Item index
                    clicked_item_idx = rel_y - header_offset
                    
                    if 0 <= clicked_item_idx < len(visible_items):
                        current_idx = start_idx + clicked_item_idx
                        break  # Select!
                    
                    # Handle Up indicator click
                    if start_idx > 0 and rel_y == 0:
                        current_idx = max(0, current_idx - page_size)
                        continue
                    
                    # Handle Down indicator click
                    footer_row = len(visible_items) + header_offset
                    if end_idx < len(items) and rel_y == footer_row:
                        current_idx = min(len(items)-1, current_idx + page_size)
                        continue
                continue
            
            # Keyboard navigation
            if key == 'UP' or key == 'k':
                current_idx = (current_idx - 1) % len(items)
            elif key == 'DOWN' or key == 'j':
                current_idx = (current_idx + 1) % len(items)
            elif key == 'HOME' or key == 'g':
                current_idx = 0
                start_idx = 0
            elif key == 'END' or key == 'G':
                current_idx = len(items) - 1
            elif key == 'PAGE_UP' or key == '[':
                current_idx = max(0, current_idx - page_size)
            elif key == 'PAGE_DOWN' or key == ']':
                current_idx = min(len(items) - 1, current_idx + page_size)
            elif key == 'ENTER':
                break
            elif key == '0' and allow_back:
                current_idx = len(items) - 1
                break
            elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                num = int(key)
                if 1 <= num <= len(options):
                    current_idx = num - 1
                    continue
            elif key == 'ESC' and allow_back:
                # Return None for back
                print(RESET + "\n" * max_view_lines + "\r")
                if sys.stdout.isatty():
                    print("\033[?1000l\033[?1006l", end="", flush=True)
                if os.name != 'nt':
                    restore_mode(fd_raw, old_raw)
                print(SHOW_CURSOR, end="")
                return None
                
    except KeyboardInterrupt:
        print(RESET + "\n" * max_view_lines + "\r")
        if sys.stdout.isatty():
            print("\033[?1000l\033[?1006l", end="", flush=True)
        if os.name != 'nt':
            restore_mode(fd_raw, old_raw)
        print(SHOW_CURSOR, end="")
        return None
    finally:
        # Restore cursor & mouse
        print(RESET + "\n" * max_view_lines + "\r")
        if sys.stdout.isatty():
            print("\033[?1000l\033[?1006l", end="", flush=True)
        if os.name != 'nt':
            restore_mode(fd_raw, old_raw)
        print(SHOW_CURSOR, end="")

    selected_label, selected_val = items[current_idx]
    return selected_val


def prompt_choice(prompt, options, allow_back=True, default_index=0):
    """Simplified choice prompt using menu system."""
    return prompt_menu(prompt, options, allow_back=allow_back, default_index=default_index)


def prompt_text(prompt, default=None, required=True):
    """Get text input from user."""
    try:
        import readline  # Enable arrow key support
    except ImportError:
        pass
    
    default_str = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{prompt}{default_str}: ").strip()
            if not value and default:
                return default
            if not value and required:
                print("   This field is required.")
                continue
            return value
        except KeyboardInterrupt:
            return None


def browse_files(start_dir="."):
    """Interactively browse file system and return selected file path."""
    current_dir = os.path.abspath(start_dir)
    
    while True:
        try:
            items = os.listdir(current_dir)
        except PermissionError:
            print(f"❌ Permission denied: {current_dir}")
            current_dir = os.path.dirname(current_dir)
            continue
            
        dirs = []
        files = []
        for item in items:
            if item.startswith('.'): continue
            full_path = os.path.join(current_dir, item)
            if os.path.isdir(full_path):
                dirs.append(item)
            else:
                files.append(item)
        
        dirs.sort()
        files.sort()
        
        menu_items = []
        if os.path.dirname(current_dir) != current_dir:
            menu_items.append(("📂 .. (Up Directory)", ".."))
        
        for d in dirs:
            menu_items.append((f"📁 {d}/", d))
        for f in files:
            menu_items.append((f"📄 {f}", f))
        
        print(f"\n📂 Location: {current_dir}")
        choice = prompt_choice(None, menu_items, allow_back=True)
        
        if choice is None:
            return None
        
        if choice == "..":
            current_dir = os.path.dirname(current_dir)
        else:
            selected_path = os.path.join(current_dir, choice)
            if os.path.isdir(selected_path):
                current_dir = selected_path
            else:
                return selected_path


def prompt_file(prompt, must_exist=True):
    """Get file path input from user with browsing support."""
    while True:
        if must_exist:
            print(f"\n{prompt}")
            options = [
                ("📂 Browse Files", "browse"),
                ("⌨️  Enter Path Manually", "manual"),
                ("🔙 Cancel", None)
            ]
            method = prompt_choice(None, options, allow_back=False)
            
            if method is None:
                return None
            
            if method == "browse":
                path = browse_files()
                if path:
                    return path
                continue
        
        try:
            path = input(f"Enter file path: ").strip()
            if not path:
                print("   This field is required.")
                continue
            if must_exist and not os.path.exists(path):
                print(f"   File not found: {path}")
                continue
            return path
        except KeyboardInterrupt:
            return None


def check_overwrite(filepath, always_overwrite=False, never_overwrite=False, is_batch=False):
    """Check if file exists and prompt user.
    
    Returns: (should_write, final_path, always_overwrite, never_overwrite)
    """
    if never_overwrite:
        return False, filepath, False, True
        
    if not os.path.exists(filepath) or always_overwrite:
        return True, filepath, always_overwrite, False
        
    print(f"\n⚠️  File already exists: {filepath}")
    
    options = [
        ("Yes", "y"),
    ]
    
    if is_batch:
        options.append(("No (skip file)", "n"))
        options.append(("Always (overwrite all remaining)", "a"))
        options.append(("Never (skip all remaining)", "v"))
    else:
        options.append(("No", "n"))
        
    options.append(("Rename (auto-increment)", "r"))
    
    choice = prompt_choice("Overwrite?", options)
    
    if choice == "y":
        return True, filepath, False, False
    elif choice == "a":
        return True, filepath, True, False
    elif choice == "v":
        print(f"⏭️  Skipping {os.path.basename(filepath)} (and all remaining)")
        return False, filepath, False, True
    elif choice == "r":
        base, ext = os.path.splitext(filepath)
        counter = 1
        new_path = f"{base}_{counter}{ext}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base}_{counter}{ext}"
        print(f"📝 Renaming to: {new_path}")
        return True, new_path, False, False
    elif choice is None:
        print("❌ Operation cancelled.")
        return False, None, False, False
    else:
        print(f"⏭️  Skipping {os.path.basename(filepath)}")
        return False, filepath, False, False


def wait_for_back(prompt="\nExecution Complete"):
    """Show a simple 'Back' menu instead of just waiting for Enter.
    
    This provides a mouse-clickable Back button and consistent navigation.
    """
    print() # Spacer
    prompt_menu(prompt, [], allow_back=True)
