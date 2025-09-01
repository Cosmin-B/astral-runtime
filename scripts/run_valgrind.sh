#!/bin/bash
# run_valgrind.sh - Valgrind Memcheck Validation
#
# Purpose: Detect memory errors and allocations in hot paths
# Usage: ./scripts/run_valgrind.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/dev"
TEST_BIN="$BUILD_DIR/tests/test_memory_validation"
LOG_FILE="$PROJECT_ROOT/valgrind_memcheck.log"

echo "=== Astral Valgrind Memcheck Validation ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "Test binary: $TEST_BIN"
echo ""

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "[ERROR] Valgrind is not installed"
    echo "Install with: sudo apt-get install valgrind (Ubuntu/Debian)"
    echo "            or: sudo yum install valgrind (RHEL/CentOS)"
    echo "            or: brew install valgrind (macOS)"
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
echo "Step 2: Running Valgrind memcheck..."
echo "This may take 10-20x longer than normal execution..."
echo ""

valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file="$LOG_FILE" \
    "$TEST_BIN"

echo ""
echo "=== Valgrind Memcheck Results ==="
echo ""

# Check for errors
if grep -q "ERROR SUMMARY: 0 errors" "$LOG_FILE"; then
    echo "[PASS] No memory errors detected"
else
    echo "[FAIL] Memory errors detected:"
    grep "ERROR SUMMARY" "$LOG_FILE"
    echo ""
    echo "See full report: $LOG_FILE"
fi

# Check for leaks
if grep -q "definitely lost: 0 bytes" "$LOG_FILE"; then
    echo "[PASS] No memory leaks detected"
else
    echo "[FAIL] Memory leaks detected:"
    grep "definitely lost" "$LOG_FILE"
    echo ""
    echo "See full report: $LOG_FILE"
fi

# Check for invalid reads/writes
echo ""
echo "=== Invalid Memory Access ==="
if grep -q "Invalid read" "$LOG_FILE"; then
    echo "[FAIL] Invalid reads detected:"
    grep -c "Invalid read" "$LOG_FILE" || echo "0"
else
    echo "[PASS] No invalid reads"
fi

if grep -q "Invalid write" "$LOG_FILE"; then
    echo "[FAIL] Invalid writes detected:"
    grep -c "Invalid write" "$LOG_FILE" || echo "0"
else
    echo "[PASS] No invalid writes"
fi

# Check for uninitialized values
if grep -q "Conditional jump or move depends on uninitialised value" "$LOG_FILE"; then
    echo "[FAIL] Uninitialized value usage detected"
    grep -c "Conditional jump or move depends on uninitialised value" "$LOG_FILE" || echo "0"
else
    echo "[PASS] No uninitialized value usage"
fi

# Analyze malloc calls in hot path
echo ""
echo "=== Hot Path Allocation Analysis ==="
echo "Searching for malloc/free calls during decode and stream read..."

# This is a heuristic - Valgrind doesn't directly tell us which phase allocations occur
# We rely on the test binary's output to determine this
if grep -q "Decode hot path: ZERO allocations" "$LOG_FILE" 2>/dev/null; then
    echo "[PASS] Decode hot path has zero allocations"
else
    echo "[INFO] Check test output for decode hot path allocation count"
fi

if grep -q "Stream read hot path: ZERO allocations" "$LOG_FILE" 2>/dev/null; then
    echo "[PASS] Stream read hot path has zero allocations"
else
    echo "[INFO] Check test output for stream read hot path allocation count"
fi

echo ""
echo "=== Full Report ==="
echo "Saved to: $LOG_FILE"
echo ""
echo "To view full report:"
echo "  cat $LOG_FILE"
echo ""
echo "To view leak details:"
echo "  grep -A 20 'LEAK SUMMARY' $LOG_FILE"
echo ""
echo "To view error details:"
echo "  grep -A 10 'ERROR SUMMARY' $LOG_FILE"
