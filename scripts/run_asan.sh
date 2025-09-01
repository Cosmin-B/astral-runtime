#!/bin/bash
# run_asan.sh - AddressSanitizer Validation
#
# Purpose: Detect memory errors, leaks, and undefined behavior
# Usage: ./scripts/run_asan.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/asan"
TEST_BIN="$BUILD_DIR/tests/test_memory_validation"

echo "=== Astral AddressSanitizer Validation ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo ""

echo "Step 1: Building with AddressSanitizer and UndefinedBehaviorSanitizer..."
cd "$PROJECT_ROOT"

# Create ASAN build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address -fsanitize=undefined" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address -fsanitize=undefined" \
    -DASTRAL_BUILD_TESTS=ON \
    -DASTRAL_BUILD_BENCHMARKS=OFF

cmake --build . --target test_memory_validation -j8

if [ ! -f "$TEST_BIN" ]; then
    echo "[ERROR] Test binary not found: $TEST_BIN"
    exit 1
fi

echo ""
echo "Step 2: Running with AddressSanitizer..."
echo ""

# Configure ASAN options
export ASAN_OPTIONS="detect_leaks=1:check_initialization_order=1:strict_init_order=1:detect_stack_use_after_return=1:halt_on_error=0"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"

# Run test
set +e
"$TEST_BIN"
EXIT_CODE=$?
set -e

echo ""
echo "=== AddressSanitizer Results ==="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "[PASS] No ASAN errors detected"
    echo "[PASS] No UBSAN errors detected"
    echo ""
    echo "Test completed successfully with sanitizers enabled"
else
    echo "[FAIL] ASAN/UBSAN detected errors"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common ASAN error types:"
    echo "  - heap-buffer-overflow: Writing past allocated memory"
    echo "  - heap-use-after-free: Accessing freed memory"
    echo "  - stack-buffer-overflow: Stack buffer overrun"
    echo "  - global-buffer-overflow: Global buffer overrun"
    echo "  - memory leak: Allocated memory not freed"
    echo ""
    echo "Common UBSAN error types:"
    echo "  - signed integer overflow"
    echo "  - shift out of bounds"
    echo "  - division by zero"
    echo "  - null pointer dereference"
    echo ""
fi

echo ""
echo "=== Summary ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "All sanitizer checks passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Run Valgrind memcheck: ./scripts/run_valgrind.sh"
    echo "  2. Run Valgrind massif: ./scripts/run_massif.sh"
else
    echo "Sanitizer errors detected. Fix issues and re-run."
    echo ""
    echo "Debugging tips:"
    echo "  - Look for stack traces in the output above"
    echo "  - ASAN provides exact line numbers for errors"
    echo "  - Use 'export ASAN_OPTIONS=symbolize=1' for better symbols"
fi

exit $EXIT_CODE
