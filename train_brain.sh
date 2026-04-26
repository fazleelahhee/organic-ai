#!/bin/bash
# OrganicAI Brain Training Script
# Runs continuously, teaching the brain through Claude
# The brain learns via Hebbian weights — no strings stored

API="http://localhost:3000/api/message"
LOG="/tmp/brain_training.log"
ROUND=0

teach() {
    local question="$1"
    local answer="$2"
    # First call teaches (Claude answers, brain stores in weights)
    curl -s --max-time 30 -X POST "$API" -H "Content-Type: application/json" \
        -d "{\"text\":\"$question\"}" > /dev/null 2>&1
    # Second call tests recall
    local recalled=$(curl -s --max-time 30 -X POST "$API" -H "Content-Type: application/json" \
        -d "{\"text\":\"$question\"}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message','FAIL'))" 2>/dev/null)
    echo "  $question → $recalled" >> "$LOG"
}

teach_from_claude() {
    local prompt="$1"
    local result=$(claude -p "$prompt" --output-format text 2>/dev/null)
    if [ -n "$result" ]; then
        # Parse lines of "Q: ... A: ..." format
        echo "$result" | while IFS= read -r line; do
            if echo "$line" | grep -q "^Q:"; then
                q=$(echo "$line" | sed 's/^Q: *//')
                read -r aline
                a=$(echo "$aline" | sed 's/^A: *//')
                if [ -n "$q" ] && [ -n "$a" ]; then
                    curl -s --max-time 30 -X POST "$API" -H "Content-Type: application/json" \
                        -d "{\"text\":\"$q\"}" > /dev/null 2>&1
                    echo "  Taught: $q → $a" >> "$LOG"
                fi
            fi
        done
    fi
}

test_brain() {
    echo "" >> "$LOG"
    echo "=== TEST (Round $ROUND) $(date) ===" >> "$LOG"
    local pass=0
    local total=0
    for q in "What is the capital of Japan?" "Who wrote Hamlet?" "What is 2+3?" "What is DNA?" "What is the speed of light?"; do
        local r=$(curl -s --max-time 30 -X POST "$API" -H "Content-Type: application/json" \
            -d "{\"text\":\"$q\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','FAIL'))" 2>/dev/null)
        total=$((total + 1))
        if [ -n "$r" ] && [ "$r" != "FAIL" ] && [ ${#r} -gt 1 ]; then
            pass=$((pass + 1))
        fi
        echo "  TEST: $q → $r" >> "$LOG"
    done
    echo "  Score: $pass/$total" >> "$LOG"
}

echo "=== BRAIN TRAINING STARTED $(date) ===" > "$LOG"
echo "Target: teach math, language, knowledge, conversation" >> "$LOG"

while true; do
    ROUND=$((ROUND + 1))
    echo "" >> "$LOG"
    echo "=== ROUND $ROUND — $(date) ===" >> "$LOG"

    # === PHASE 1: MATH (basics to complex) ===
    if [ $ROUND -le 5 ]; then
        echo "  Phase: Basic Math" >> "$LOG"
        claude -p "Generate 20 math questions with answers, from basic to intermediate. Format each as two lines:
Q: [question]
A: [answer]
Examples: Q: What is 7+5? A: 12. Include addition, subtraction, multiplication, division, fractions, percentages." --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === PHASE 2: ALPHABET & WORDS ===
    if [ $ROUND -le 10 ]; then
        echo "  Phase: Letters & Words" >> "$LOG"
        for letter in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z; do
            teach "What letter is $letter?" "$letter is a letter of the English alphabet"
        done
        claude -p "Generate 30 common English words with their definitions. Format:
Q: What does [word] mean?
A: [definition]
Include simple words (happy, water, sun) and harder ones (eloquent, perseverance, ambiguous)." --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === PHASE 3: KNOWLEDGE ===
    echo "  Phase: Knowledge" >> "$LOG"
    claude -p "Generate 20 general knowledge questions with short answers. Format:
Q: [question]
A: [answer]
Cover: science, history, geography, literature, technology, nature, arts, music. Different topics each time. Be concise." --output-format text 2>/dev/null | while IFS= read -r line; do
        q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
        if [ -n "$q" ]; then
            teach "$q" ""
        fi
    done

    # === PHASE 4: CAPITAL CITIES ===
    if [ $ROUND -le 3 ]; then
        echo "  Phase: Capital Cities" >> "$LOG"
        claude -p "List 50 countries and their capital cities. Format each as:
Q: What is the capital of [country]?
A: [capital]" --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === PHASE 5: CONVERSATION ===
    echo "  Phase: Conversation" >> "$LOG"
    claude -p "Generate 15 conversational question-answer pairs that teach an AI how to have natural conversations. Include greetings, personal questions, opinions, emotions, humor. Format:
Q: [question]
A: [natural response]
Make responses warm, helpful, and human-like." --output-format text 2>/dev/null | while IFS= read -r line; do
        q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
        if [ -n "$q" ]; then
            teach "$q" ""
        fi
    done

    # === PHASE 6: SENTENCE FORMATION ===
    if [ $ROUND -le 8 ]; then
        echo "  Phase: Sentence Formation" >> "$LOG"
        claude -p "Generate 15 examples teaching sentence structure. Format:
Q: Make a sentence with the word [word]
A: [example sentence]
Use diverse words: nouns, verbs, adjectives. Show proper grammar." --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === PHASE 7: CREATIVE WRITING ===
    if [ $ROUND -ge 3 ]; then
        echo "  Phase: Creative Writing" >> "$LOG"
        claude -p "Generate 10 creative writing examples. Include:
- 3 short poems (2-4 lines each)
- 3 story openings (1-2 sentences)
- 2 business proposal sentences
- 2 descriptive paragraphs (1-2 sentences)
Format:
Q: Write a poem about [topic]
A: [poem]
Q: Write a story opening about [topic]
A: [opening]" --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === PHASE 8: DICTIONARY ===
    echo "  Phase: Dictionary" >> "$LOG"
    claude -p "Generate 30 English words with meanings. Each word should be different from previous rounds. Include:
- 10 common words
- 10 intermediate words
- 10 advanced/rare words
Format:
Q: Define [word]
A: [definition]" --output-format text 2>/dev/null | while IFS= read -r line; do
        q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
        if [ -n "$q" ]; then
            teach "$q" ""
        fi
    done

    # === PHASE 9: ADVANCED MATH ===
    if [ $ROUND -ge 5 ]; then
        echo "  Phase: Advanced Math" >> "$LOG"
        claude -p "Generate 15 math questions covering algebra, geometry, and basic calculus concepts. Format:
Q: [question]
A: [answer with brief explanation]
Include: solve equations, area/perimeter, basic derivatives, simple proofs." --output-format text 2>/dev/null | while IFS= read -r line; do
            q=$(echo "$line" | grep "^Q:" | sed 's/^Q: *//')
            if [ -n "$q" ]; then
                teach "$q" ""
            fi
        done
    fi

    # === TEST every 3 rounds ===
    if [ $((ROUND % 3)) -eq 0 ]; then
        test_brain
    fi

    echo "  Round $ROUND complete — $(date)" >> "$LOG"
    echo "Round $ROUND complete"

    # Brief pause between rounds
    sleep 5
done
