#!/bin/bash
# run_massif.sh - Valgrind Massif Heap Profiler
#
# Purpose: Profile heap growth to detect allocations in hot paths
# Usage: ./scripts/run_massif.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/dev"
TEST_BIN="$BUILD_DIR/tests/test_memory_validation"
MASSIF_OUT="$PROJECT_ROOT/massif.out"
MASSIF_REPORT="$PROJECT_ROOT/massif_report.txt"

echo "=== Astral Valgrind Massif Heap Profiler ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "Test binary: $TEST_BIN"
echo ""

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "[ERROR] Valgrind is not installed"
    exit 1
fi

echo "Step 1: Building with debug symbols..."
cd "$PROJECT_ROOT"
cmake --preset dev -DCMAKE_BUILD_TYPE=Debug
cmake --build build/dev --target test_memory_validation -j8

if [ ! -f "$TEST_BIN" ]; then
    echo "[ERROR] Test binary not found: $TEST_BIN"
    exit 1
fi

echo ""
echo "Step 2: Running Valgrind massif..."
echo "This profiles heap usage over time..."
echo ""

valgrind \
    --tool=massif \
    --massif-out-file="$MASSIF_OUT" \
    --detailed-freq=1 \
    --max-snapshots=100 \
    --threshold=0.1 \
    --time-unit=ms \
    "$TEST_BIN"

echo ""
echo "Step 3: Generating massif report..."

if ! command -v ms_print &> /dev/null; then
    echo "[ERROR] ms_print not found (part of Valgrind)"
    echo "Skipping report generation"
    exit 1
fi

ms_print "$MASSIF_OUT" > "$MASSIF_REPORT"

echo ""
echo "=== Massif Heap Profile Results ==="
echo ""

# Extract peak memory usage
if grep -q "peak" "$MASSIF_REPORT"; then
    echo "Peak heap usage:"
    grep "peak" "$MASSIF_REPORT" | head -5
else
    echo "[INFO] No peak information found"
fi

echo ""
echo "=== Heap Growth Analysis ==="

# Look for growth during hot paths
# This is heuristic - we're looking for sudden increases in heap usage
echo "Total snapshots taken:"
grep -c "^#" "$MASSIF_REPORT" || echo "0"

echo ""
echo "Memory timeline (first 20 snapshots):"
grep "^#" "$MASSIF_REPORT" | head -20

echo ""
echo "=== Hot Path Validation ==="
echo ""
echo " Heap should NOT grow during decode loop or stream read"
echo "If heap grows during these phases, it indicates allocations in hot path"
echo ""
echo "Check the timeline above for:"
echo "  1. Initial growth during initialization (EXPECTED)"
echo "  2. Stable heap during decode iterations (REQUIRED)"
echo "  3. Stable heap during stream read iterations (REQUIRED)"
echo "  4. Decline during shutdown (EXPECTED)"

echo ""
echo "=== Full Reports ==="
echo "Massif output: $MASSIF_OUT"
echo "Human-readable report: $MASSIF_REPORT"
echo ""
echo "To view full report:"
echo "  cat $MASSIF_REPORT"
echo ""
echo "To view in less:"
echo "  less $MASSIF_REPORT"
echo ""
echo "To see peak snapshot details:"
echo "  grep -A 30 'peak' $MASSIF_REPORT"
