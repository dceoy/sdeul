#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-us.anthropic.claude-sonnet-4-20250514-v1:0}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --anthropic-model\"" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --anthropic-model\" with output file" {
  local output_file="/tmp/test_anthropic_output.json"
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --anthropic-model\" with compact JSON" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --anthropic-model\" with skip validation" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --anthropic-model\" with custom temperature" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --anthropic-model\" with max tokens" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --anthropic-model\" with debug flag" {
  run uv run sdeul extract \
    --anthropic-model="${ANTHROPIC_MODEL}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
