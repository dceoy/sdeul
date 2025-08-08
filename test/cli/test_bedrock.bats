#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  aws sts get-caller-identity
  export BEDROCK_MODEL="${BEDROCK_MODEL:-us.anthropic.claude-sonnet-4-20250514-v1:0}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --bedrock-model\"" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --bedrock-model\" with output file" {
  local output_file="/tmp/test_bedrock_output.json"
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --bedrock-model\" with compact JSON" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --bedrock-model\" with skip validation" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --bedrock-model\" with custom temperature" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --bedrock-model\" with max tokens" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --bedrock-model\" with debug flag" {
  run uv run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
