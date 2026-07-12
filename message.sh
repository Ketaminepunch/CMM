```bash
#!/bin/bash

# check_moulinette.sh - Helper script to run the moulinette tests

MOULINETTE_DIR="$HOME/Downloads/moulinette"
PROJECT_DIR="$(pwd)"

if [ ! -d "$MOULINETTE_DIR" ]; then
    echo "Error: Moulinette directory not found at $MOULINETTE_DIR"
    exit 1
fi

echo "=================================================="
# 1. Sync Moulinette Dependencies
echo "Syncing moulinette dependencies..."
(cd "$MOULINETTE_DIR" && uv sync)
echo "=================================================="

# ---- PUBLIC EVALUATION ----
echo ""
echo "=================================================="
echo "RUNNING PUBLIC EVALUATION"
echo "=================================================="
# Generate public exercises
(cd "$MOULINETTE_DIR" && uv run python -m moulinette prepare_exercises --set public)

# Run student project against public exercises
uv run python -m src \
  --functions_definition "$MOULINETTE_DIR/data/input/functions_definition.json" \
  --input "$MOULINETTE_DIR/data/input/function_calling_tests.json" \
  --output "$PROJECT_DIR/data/output/public_function_calls.json"

# Grade public answers
(cd "$MOULINETTE_DIR" && uv run python -m moulinette grade_student_answers --set public --student_answer_path "$PROJECT_DIR/data/output/public_function_calls.json")


# ---- PRIVATE EVALUATION ----
echo ""
echo "=================================================="
echo "RUNNING PRIVATE EVALUATION"
echo "=================================================="
# Generate private exercises
(cd "$MOULINETTE_DIR" && uv run python -m moulinette prepare_exercises --set private)

# Run student project against private exercises
uv run python -m src \
  --functions_definition "$MOULINETTE_DIR/data/input/functions_definition.json" \
  --input "$MOULINETTE_DIR/data/input/function_calling_tests.json" \
  --output "$PROJECT_DIR/data/output/private_function_calls.json"

# Grade private answers
(cd "$MOULINETTE_DIR" && uv run python -m moulinette grade_student_answers --set private --student_answer_path "$PROJECT_DIR/data/output/private_function_calls.json")
```