import os

# các folder cần bỏ qua
IGNORE_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache"
}

def print_tree(folder, indent=""):
    dirs = sorted([
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
        and d not in IGNORE_DIRS
        and not d.startswith(".")
    ])

    for i, d in enumerate(dirs):
        path = os.path.join(folder, d)

        if i == len(dirs) - 1:
            connector = "└── "
            new_indent = indent + "    "
        else:
            connector = "├── "
            new_indent = indent + "│   "

        print(indent + connector + d)
        print_tree(path, new_indent)


root_path = "."

print(root_path)
print_tree(root_path)