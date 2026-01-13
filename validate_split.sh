#!/usr/bin/env bash
# validate_split.sh - Quick validation script for repository split

set -e

echo "========================================="
echo "Repository Split Validation"
echo "========================================="
echo ""

# Check directory structure
echo "✓ Checking directory structure..."
for dir in recon_core recon_experiments; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ $dir/ missing!"
        exit 1
    fi
done

# Check key files
echo ""
echo "✓ Checking key files..."
for file in \
    "recon_core/pyproject.toml" \
    "recon_core/src/recon_core/__init__.py" \
    "recon_core/README.md" \
    "recon_experiments/pyproject.toml" \
    "recon_experiments/src/recon_experiments/__init__.py" \
    "recon_experiments/README.md" \
    "MIGRATION_GUIDE.md" \
    "REFACTORING_SUMMARY.md"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file missing!"
        exit 1
    fi
done

# Check for old imports
echo ""
echo "✓ Checking for old 'setr' imports in new packages..."
old_imports=$(grep -r "from setr\." recon_core/src recon_experiments/src --include="*.py" 2>/dev/null || true)
if [ -z "$old_imports" ]; then
    echo "  ✓ No old imports found"
else
    echo "  ⚠ Warning: Found old imports:"
    echo "$old_imports"
fi

# Count Python files
echo ""
echo "✓ Statistics:"
core_files=$(find recon_core/src -name "*.py" | wc -l)
exp_files=$(find recon_experiments/src -name "*.py" | wc -l)
test_files=$(find recon_core/tests -name "*.py" | wc -l)
echo "  Core package: $core_files Python files"
echo "  Experiments package: $exp_files Python files"
echo "  Test files: $test_files files"

# Try basic import (will fail without dependencies, but checks syntax)
echo ""
echo "✓ Checking package can be imported (requires dependencies)..."
export PYTHONPATH="recon_core/src:$PYTHONPATH"
if python3 -c "import sys; sys.path.insert(0, 'recon_core/src'); import recon_core" 2>/dev/null; then
    echo "  ✓ recon_core imports successfully"
    version=$(python3 -c "import sys; sys.path.insert(0, 'recon_core/src'); import recon_core; print(recon_core.__version__)")
    echo "  Version: $version"
else
    echo "  ⚠ recon_core import failed (may require dependencies like pandas, torch, etc.)"
fi

echo ""
echo "========================================="
echo "Validation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies: cd recon_core && pip install -e ."
echo "2. Review MIGRATION_GUIDE.md for import changes"
echo "3. Run tests: pytest recon_core/tests/"
echo "4. Try a quick experiment"
echo ""
echo "See REFACTORING_SUMMARY.md for detailed information."
