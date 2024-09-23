#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  MODEL_DIR_PATH='./test/model'
  [[ -d "${MODEL_DIR_PATH}" ]] || mkdir "${MODEL_DIR_PATH}"
  LLAMA31_8B_IT_GGUF_URL='https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf'
  export LLAMA31_8B_IT_GGUF_FILE_PATH="${MODEL_DIR_PATH%/}/${LLAMA31_8B_IT_GGUF_URL##*/}"
  [[ -f "${LLAMA31_8B_IT_GGUF_FILE_PATH}" ]] || curl -SL -o "${LLAMA31_8B_IT_GGUF_FILE_PATH}" "${LLAMA31_8B_IT_GGUF_URL}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --model-file\" and Llama 3.1" {
  run poetry run sdeul extract \
    --model-file="${LLAMA31_8B_IT_GGUF_FILE_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
