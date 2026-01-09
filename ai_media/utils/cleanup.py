import os
import shutil
from pathlib import Path

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
