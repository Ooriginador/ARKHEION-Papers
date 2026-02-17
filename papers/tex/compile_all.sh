#!/bin/bash
set -e
OUTDIR="$(pwd)/pdf_output"
LOGDIR="$(pwd)/pdf_output/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

SUCCESS=0
FAIL=0
FAILED_LIST=""

compile_tex() {
    local texfile="$1"
    local basename=$(basename "$texfile" .tex)
    local dirname=$(dirname "$texfile")
    
    echo -n "  Compiling $basename... "
    
    cd "$dirname"
    # Run pdflatex twice for references
    if pdflatex -interaction=nonstopmode -output-directory="$OUTDIR" "$basename.tex" > "$LOGDIR/${basename}.log" 2>&1; then
        # Second pass for refs
        pdflatex -interaction=nonstopmode -output-directory="$OUTDIR" "$basename.tex" >> "$LOGDIR/${basename}.log" 2>&1
        echo "✅"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $basename"
        # Extract key error
        grep -A2 "^!" "$LOGDIR/${basename}.log" 2>/dev/null | head -6
    fi
    cd /home/jhonslife/ARKHEION_AGI_2.0/docs/papers
}

echo "═══════════════════════════════════════════"
echo "  ARKHEION Papers — PDF Compilation"
echo "═══════════════════════════════════════════"
echo ""

echo "── Level 0 ──"
for f in level_0/*.tex; do compile_tex "$f"; done

echo "── Level 1 Core ──"
for f in level_1_core/*.tex; do compile_tex "$f"; done

echo "── Level 1 AI ──"
for f in level_1_ai/*.tex; do compile_tex "$f"; done

echo "── Level 1 Data ──"
for f in level_1_data/*.tex; do compile_tex "$f"; done

echo "── Level 1 Apps ──"
for f in level_1_apps/*.tex; do compile_tex "$f"; done

echo "── Level 2 Integration ──"
for f in level_2_integration/*.tex; do compile_tex "$f"; done

echo ""
echo "═══════════════════════════════════════════"
echo "  Results: ✅ $SUCCESS success | ❌ $FAIL failed"
if [ -n "$FAILED_LIST" ]; then
    echo "  Failed:$FAILED_LIST"
fi
echo "═══════════════════════════════════════════"
