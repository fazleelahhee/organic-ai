#!/bin/bash
# SMART training — conserves Claude tokens
#
# Strategy: ONE Claude call generates a big batch of teaching material.
# Save it to a file. Feed the file to the brain line by line.
# Re-use the same file across multiple rounds — no extra Claude calls.
#
# First run: generates teaching files (costs tokens)
# All subsequent runs: re-uses files (FREE — zero tokens)

API="http://localhost:3000/api/message"
DATA_DIR="$HOME/organic-ai/training_data"
LOG="/tmp/brain_training.log"

mkdir -p "$DATA_DIR"

teach_line() {
    local q="$1"
    local a="$2"
    if [ -n "$q" ] && [ -n "$a" ]; then
        # First call: brain tries, Claude answers if needed, brain stores
        curl -s --max-time 30 -X POST "$API" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"$q\"}" > /dev/null 2>&1
        echo "  Taught: $q" >> "$LOG"
    fi
}

teach_file() {
    local file="$1"
    local label="$2"
    if [ ! -f "$file" ]; then
        echo "  File not found: $file" >> "$LOG"
        return
    fi
    echo "  Teaching from: $label ($file)" >> "$LOG"
    while IFS='|' read -r question answer; do
        if [ -n "$question" ] && [ -n "$answer" ]; then
            teach_line "$question" "$answer"
        fi
    done < "$file"
}

generate_if_missing() {
    local file="$1"
    local prompt="$2"
    local label="$3"

    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "  $label: already generated (reusing — zero tokens)" >> "$LOG"
        return
    fi

    echo "  $label: generating with Claude (one-time cost)..." >> "$LOG"
    claude -p "$prompt" --output-format text 2>/dev/null > "$file"
    echo "  $label: saved to $file" >> "$LOG"
}

echo "=== SMART TRAINING — $(date) ===" > "$LOG"

# =====================================================
# GENERATE teaching files (one-time Claude cost)
# =====================================================

generate_if_missing "$DATA_DIR/math_basic.txt" \
"Generate 100 math Q&A pairs. Format: question|answer (pipe separated, one per line, no numbering).
Include: addition, subtraction, multiplication, division from simple to complex.
Example lines:
What is 7+5?|12
What is 15 times 3?|45
What is 144 divided by 12?|12" \
"Basic Math"

generate_if_missing "$DATA_DIR/capitals.txt" \
"Generate 100 country capital Q&A pairs. Format: question|answer (pipe separated, one per line).
Example: What is the capital of Japan?|Tokyo" \
"Capital Cities"

generate_if_missing "$DATA_DIR/vocabulary.txt" \
"Generate 200 English vocabulary words with definitions. Format: question|answer (pipe separated, one per line).
Example: What does eloquent mean?|Fluent and persuasive in speaking or writing" \
"Vocabulary"

generate_if_missing "$DATA_DIR/science.txt" \
"Generate 100 science Q&A pairs covering physics, chemistry, biology, astronomy. Format: question|answer (pipe separated).
Example: What is the speed of light?|299,792,458 meters per second" \
"Science"

generate_if_missing "$DATA_DIR/history.txt" \
"Generate 80 history Q&A pairs covering world history. Format: question|answer (pipe separated).
Example: Who was the first president of the United States?|George Washington" \
"History"

generate_if_missing "$DATA_DIR/conversation.txt" \
"Generate 50 conversational Q&A pairs for natural dialogue. Format: question|answer (pipe separated).
Include greetings, opinions, feelings, humor, advice.
Example: How are you?|I'm doing well, thanks for asking! How about you?" \
"Conversation"

generate_if_missing "$DATA_DIR/grammar.txt" \
"Generate 80 grammar and sentence examples. Format: question|answer (pipe separated).
Example: Make a sentence with the word 'perseverance'|Despite many setbacks, her perseverance led to eventual success." \
"Grammar"

generate_if_missing "$DATA_DIR/coding.txt" \
"Generate 50 programming Q&A pairs covering basic concepts. Format: question|answer (pipe separated).
Example: What is a variable?|A named storage location in memory that holds a value" \
"Coding Concepts"

generate_if_missing "$DATA_DIR/poetry.txt" \
"Generate 30 short poem examples. Format: question|answer (pipe separated).
Example: Write a poem about the moon|Silver light on midnight water, shadows dance without a sound, the moon watches every daughter, every soul on hallowed ground" \
"Poetry"

generate_if_missing "$DATA_DIR/reasoning.txt" \
"Generate 50 logic and reasoning Q&A pairs. Format: question|answer (pipe separated).
Include: analogies, cause-effect, deduction, comparisons.
Example: If all dogs are animals and Rex is a dog, what is Rex?|Rex is an animal" \
"Reasoning"

echo "" >> "$LOG"
echo "=== TEACHING PHASE (zero Claude tokens) ===" >> "$LOG"

# =====================================================
# TEACH from files (zero Claude tokens — all local)
# =====================================================
ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND — $(date) ===" >> "$LOG"

    for file in "$DATA_DIR"/*.txt; do
        label=$(basename "$file" .txt)
        teach_file "$file" "$label"
    done

    # Test every 5 rounds
    if [ $((ROUND % 5)) -eq 0 ]; then
        echo "" >> "$LOG"
        echo "  === TEST ROUND $ROUND ===" >> "$LOG"
        for q in "What is 7+5?" "What is the capital of France?" "What does eloquent mean?" "Who wrote Hamlet?"; do
            r=$(curl -s --max-time 30 -X POST "$API" -H "Content-Type: application/json" \
                -d "{\"text\":\"$q\"}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message','FAIL'))" 2>/dev/null)
            echo "  TEST: $q → $r" >> "$LOG"
        done
    fi

    echo "  Round $ROUND complete — $(date)" >> "$LOG"
    sleep 2
done
