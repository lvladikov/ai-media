import os
import shutil
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """
    Format bytes into human-readable size (B, KB, MB, GB, TB).
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string like "1.5 GB" or "256 KB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    elif size_bytes < 1024 ** 4:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    else:
        return f"{size_bytes / (1024 ** 4):.1f} TB"


def get_folder_size(path: str) -> int:
    """
    Recursively calculate the total size of a folder.
    
    Args:
        path: Path to the folder
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size


def clear_directory(path: str) -> list[str]:
    """
    Clear all files and subdirectories in the given directory.
    Skips hidden files (starting with .).
    
    Args:
        path (str): Path to the directory to clear.
        
    Returns:
        list[str]: List of names of deleted items.
    """
    if not os.path.exists(path):
        return []
    
    deleted_items = []
    try:
        items = os.listdir(path)
    except OSError:
        return []
        
    for item in items:
        # Skip hidden files (like .gitkeep or .DS_Store)
        if item.startswith("."):
            continue
            
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                deleted_items.append(item)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                # Mark as directory in output
                deleted_items.append(f"{item}/")
        except Exception as e:
            print(f"❌ Failed to delete {item_path}: {e}")
            
    return deleted_items
