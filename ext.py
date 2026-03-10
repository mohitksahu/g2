import os
import re

# Set your target directory here
MAIN_PATH = r"F:/LabBasedProject/g2"

# Folders to completely ignore
IGNORE_FOLDERS = {
    "venv", ".git", ".vscode", "node_modules", "dist", "build",
    ".angular", "coverage", ".idea", ".vs", "__pycache__", ".pytest_cache", "data", "dataset"
}

# Files to ignore by exact name
IGNORE_FILES = {
    "curr_dir.txt", "package-lock.json", ".DS_Store", "thumbs.db",
    ".editorconfig", "package.json", "angular.json", ".gitignore", "Dockerfile"
}

# Extensions to ignore
IGNORE_EXTENSIONS = {
    ".map", ".log", ".lock", ".jpg", ".jpeg", ".png", ".gif", ".ico",
    ".svg", ".woff", ".woff2", ".ttf", ".eot", ".md"
}

# Extensions to INCLUDE (whitelist)
INCLUDE_EXTENSIONS = {
    ".py", ".ts", ".js", ".html", ".css", ".scss", ".json", ".md", ".txt"
}


def remove_comments(content, file_ext):
    """Remove comments from code based on file extension. Also removes excessive blank lines."""
    if file_ext in {".py"}:
        # Remove Python comments (# ...) but preserve shebang
        lines = content.split('\n')
        result = []
        for i, line in enumerate(lines):
            # Keep shebang on first line
            if i == 0 and line.startswith('#!'):
                result.append(line)
                continue
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue  # Skip full-line comments
            if '#' in line:
                parts = line.split('#', 1)
                if parts[0].strip():
                    result.append(parts[0].rstrip())
                    continue
            result.append(line)
        content = '\n'.join(result)
    elif file_ext in {".js", ".ts", ".css", ".scss"}:
        # Remove multi-line comments /* ... */
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        # Remove single-line comments // ...
        lines = content.split('\n')
        result = []
        for line in lines:
            if '//' in line:
                if 'http://' not in line and 'https://' not in line:
                    line = re.sub(r'//.*$', '', line)
            if line.strip():
                result.append(line)
        content = '\n'.join(result)
    elif file_ext == ".html":
        # Remove HTML comments <!-- ... -->
        content = re.sub(r'<!--[\s\S]*?-->', '', content)
    elif file_ext == ".json":
        # Remove // style comments (common in JSON-like configs)
        lines = content.split('\n')
        result = [line for line in lines if not line.strip().startswith('//')]
        content = '\n'.join(result)
    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)
    # Remove trailing whitespace from lines
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    return content.strip()


def should_include_file(file_name, file_path):
    """Determine if a file should be included"""
    if file_name in IGNORE_FILES:
        return False
    _, ext = os.path.splitext(file_name)
    ext_lower = ext.lower()
    if ext_lower in IGNORE_EXTENSIONS:
        return False
    if ext_lower in INCLUDE_EXTENSIONS:
        return True
    important_files = {
        "Dockerfile", "docker-compose.yml", ".env.example", "README"
    }
    return file_name in important_files


def get_relative_path(file_path, root_dir):
    """Get clean relative path with forward slashes"""
    rel_path = os.path.relpath(file_path, root_dir)
    return rel_path.replace(os.sep, '/')


def write_directory_contents(root_dir):
    """Create a comprehensive text file with directory tree and file contents"""
    if not os.path.exists(root_dir):
        print(f"❌ Directory not found: {root_dir}")
        return

    output_file = os.path.join(root_dir, "curr_dir.txt")
    # Collect all files
    all_files = []
    for folder, subdirs, files in os.walk(root_dir):
        subdirs[:] = [d for d in subdirs if d not in IGNORE_FOLDERS]
        for file in files:
            if should_include_file(file, os.path.join(folder, file)):
                all_files.append(os.path.join(folder, file))
    all_files.sort()
    print(f"📊 Found {len(all_files)} files to include")

    with open(output_file, "w", encoding="utf-8") as out:
        # PART 1: DIRECTORY TREE
        out.write("DIRECTORY TREE\n")
        for folder, subdirs, files in os.walk(root_dir):
            subdirs[:] = [d for d in subdirs if d not in IGNORE_FOLDERS]
            subdirs.sort()
            level = folder.replace(root_dir, '').count(os.sep)
            indent = '│ ' * level
            folder_name = os.path.basename(folder) or os.path.basename(root_dir)
            out.write(f"{indent}📁 {folder_name}/\n")
            sub_indent = '│ ' * (level + 1)
            included_files = [f for f in files if should_include_file(f, os.path.join(folder, f))]
            included_files.sort()
            for file in included_files:
                icon = "📄"
                if file.endswith('.ts'):
                    icon = "📘"
                elif file.endswith('.html'):
                    icon = "🌐"
                elif file.endswith(('.css', '.scss')):
                    icon = "🎨"
                elif file.endswith('.json'):
                    icon = "⚙️"
                elif file.endswith('.md'):
                    icon = "📝"
                out.write(f"{sub_indent}{icon} {file}\n")

        # PART 2: FILE CONTENTS (without comments)
        out.write("FILE CONTENTS\n")
        original_size = 0
        processed_size = 0
        for file_path in all_files:
            rel_path = get_relative_path(file_path, root_dir)
            out.write(f"FILE: {rel_path}\n")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    original_size += len(content)
                if not content.strip():
                    out.write("[EMPTY FILE]\n")
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                processed_content = remove_comments(content, ext)
                processed_size += len(processed_content)
                if not processed_content.strip():
                    out.write("[FILE CONTAINS ONLY COMMENTS]\n")
                else:
                    out.write(processed_content)
                    if not processed_content.endswith('\n'):
                        out.write('\n')
            except UnicodeDecodeError:
                out.write("[BINARY FILE]\n")
            except Exception as e:
                out.write(f"[ERROR: {e}]\n")

        # PART 3: SUMMARY
        out.write("SUMMARY\n")
        out.write(f"Total files: {len(all_files)}\n")
        if original_size > 0:
            reduction = ((original_size - processed_size) / original_size) * 100
            out.write(f"Original size: {original_size:,} chars\n")
            out.write(f"Processed size: {processed_size:,} chars\n")
            out.write(f"Reduction: {reduction:.1f}%\n")
        ext_counts = {}
        for fp in all_files:
            ext = os.path.splitext(fp)[1] or "[no extension]"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        out.write("\nFile types:\n")
        for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True):
            out.write(f" {ext}: {count}\n")

    file_size_kb = os.path.getsize(output_file) / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"📁 Size: {file_size_kb:.2f} KB")
    if original_size > 0:
        print(f"🎯 Comment reduction: {reduction:.1f}%")


def main():
    if not MAIN_PATH:
        print("❌ Please set MAIN_PATH variable")
        return
    print(f"🔍 Scanning: {MAIN_PATH}")
    print(f"📝 Ignoring: {', '.join(list(IGNORE_FOLDERS)[:5])}...")
    print(f"✅ Including: {', '.join(INCLUDE_EXTENSIONS)}")
    write_directory_contents(MAIN_PATH)


if __name__ == "__main__":
    main()