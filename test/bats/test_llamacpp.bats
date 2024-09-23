#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  LLAMACPP_MODEL_FILE_URL='https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf'
  MODEL_DIR_PATH='./test/model'
  export LLAMACPP_MODEL_FILE_PATH="${MODEL_DIR_PATH%/}/${LLAMACPP_MODEL_FILE_URL##*/}"
  [[ -d "${MODEL_DIR_PATH}" ]] || mkdir "${MODEL_DIR_PATH}"
  [[ -f "${LLAMACPP_MODEL_FILE_PATH}" ]] || curl -SL -o "${LLAMACPP_MODEL_FILE_PATH}" "${LLAMACPP_MODEL_FILE_URL}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --model-file\"" {
  run poetry run sdeul extract \
    --model-file="${LLAMACPP_MODEL_FILE_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
