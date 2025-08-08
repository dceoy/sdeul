#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export CEREBRAS_MODEL="${CEREBRAS_MODEL:-gpt-oss-120b}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --cerebras-model\"" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --cerebras-model\" with output file" {
  local output_file="/tmp/test_cerebras_output.json"
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --cerebras-model\" with compact JSON" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --cerebras-model\" with skip validation" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --cerebras-model\" with custom temperature" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --cerebras-model\" with max tokens" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --cerebras-model\" with debug flag" {
  run uv run sdeul extract \
    --cerebras-model="${CEREBRAS_MODEL}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
