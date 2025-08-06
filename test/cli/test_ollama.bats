#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export OLLAMA_MODEL="${OLLAMA_MODEL:-gpt-oss-20b}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}" # Default Ollama port
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --ollama-model\"" {
  run uv run sdeul extract \
    --ollama-model="${OLLAMA_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --ollama-model --ollama-base-url\"" {
  run uv run sdeul extract \
    --ollama-model="${OLLAMA_MODEL}" \
    --ollama-base-url="${OLLAMA_BASE_URL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
