#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GOOGLE_MODEL="${GOOGLE_MODEL:-gemini-2.5-pro}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --google-model\"" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --google-model\" with output file" {
  local output_file="/tmp/test_google_output.json"
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --google-model\" with compact JSON" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --google-model\" with skip validation" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --google-model\" with custom temperature" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --google-model\" with max tokens" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --google-model\" with debug flag" {
  run uv run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
