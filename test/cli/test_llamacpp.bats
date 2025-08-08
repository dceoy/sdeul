#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  MODEL_FILE_URL="${MODEL_FILE_URL:-https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q8_0.gguf}"
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
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with output file" {
  local output_file="/tmp/test_llamacpp_output.json"
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with compact JSON" {
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with skip validation" {
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with custom temperature" {
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with max tokens" {
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --llamacpp-model-file\" with debug flag" {
  run uv run sdeul extract \
    --llamacpp-model-file="${MODEL_FILE_PATH}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
