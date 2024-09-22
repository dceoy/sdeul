#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  MODEL_GGUF_URL='https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf'
  curl -SLO "${MODEL_GGUF_URL}"
  export MODEL_GGUF_PATH="${MODEL_GGUF_URL##*/}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --model-gguf\"" {
  run poetry run sdeul extract \
    --pretty-json \
    --model-gguf="${MODEL_GGUF_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
