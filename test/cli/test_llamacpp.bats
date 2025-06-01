#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  MODEL_FILE_URL="${MODEL_FILE_URL:-https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf}"
  if [[ -z "${MODEL_FILE_PATH:-}" ]]; then
    export MODEL_FILE_PATH="${MODEL_FILE_PATH:-./test/model/${MODEL_FILE_URL##*/}}"
  fi
  if [[ ! -f "${MODEL_FILE_PATH}" ]]; then
    [[ -d "${MODEL_FILE_PATH%/*}" ]] || mkdir -p "${MODEL_FILE_PATH%/*}"
    curl -SL -o "${MODEL_FILE_PATH}" "${MODEL_FILE_URL}" >&3
  fi
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --model-file\"" {
  run poetry run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
