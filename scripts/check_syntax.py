import ast
import pathlib
import sys

root = pathlib.Path(r"c:\Users\rijas\Desktop\Multimodal Document Intelligence")
errors = []
ok = []

for p in sorted(root.rglob("*.py")):
    if any(skip in str(p) for skip in ["venv", ".venv", "__pycache__"]):
        continue
    try:
        source = p.read_text(encoding="utf-8")
        ast.parse(source)
        ok.append(p.relative_to(root))
    except SyntaxError as e:
        errors.append((p.relative_to(root), str(e)))

for f in ok:
    print(f"  OK  {f}")

if errors:
    print("\nSYNTAX ERRORS:")
    for f, e in errors:
        print(f"  FAIL  {f}: {e}")
    sys.exit(1)
else:
    print(f"\nAll {len(ok)} files passed syntax check.")
