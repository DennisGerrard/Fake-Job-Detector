#confirming environment is ready
import sys
from pathlib import Path

print("Python", sys.version.split()[0])

ok = True
for name in ["flask", "sklearn", "pandas", "numpy", "joblib"]:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "?")
        print(f" [ok]  {name:<8} {version}")
    except ImportError:
        print(f" [MISSING] {name}")
        ok = False

for folder in ["data", "models", "templates", "static/css", "static/js"]:
    exists = Path(folder).is_dir()
    print(f"  [{'ok' if exists else 'MISSING'}]  {folder}/")
    ok = ok and exists

print("\nStage 1 passed." if ok else "\nStage 1 failed - fix the items marked MISSING.") 
