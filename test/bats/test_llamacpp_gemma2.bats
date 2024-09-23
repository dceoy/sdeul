#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  MODEL_DIR_PATH='./test/model'
  [[ -d "${MODEL_DIR_PATH}" ]] || mkdir "${MODEL_DIR_PATH}"
  GEMMA2_2B_IT_GGUF_URL='https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q6_K_L.gguf'
  export GEMMA2_2B_IT_GGUF_FILE_PATH="${MODEL_DIR_PATH%/}/${GEMMA2_2B_IT_GGUF_URL##*/}"
  [[ -f "${GEMMA2_2B_IT_GGUF_FILE_PATH}" ]] || curl -SL -o "${GEMMA2_2B_IT_GGUF_FILE_PATH}" "${GEMMA2_2B_IT_GGUF_URL}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --model-file\" and Gemma 2" {
  run poetry run sdeul extract \
    --model-file="${GEMMA2_2B_IT_GGUF_FILE_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
