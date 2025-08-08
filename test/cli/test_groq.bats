#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GROQ_MODEL="${GROQ_MODEL:-openai/gpt-oss-120b}"
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --groq-model\"" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --groq-model\" with output file" {
  local output_file="/tmp/test_groq_output.json"
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --output-json-file="${output_file}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
  [[ -f "${output_file}" ]]
  rm -f "${output_file}"
}

@test "pass with \"sdeul extract --groq-model\" with compact JSON" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --compact-json \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --groq-model\" with skip validation" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --skip-validation \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --groq-model\" with custom temperature" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --temperature=0.5 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --groq-model\" with max tokens" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --max-tokens=1000 \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --groq-model\" with debug flag" {
  run uv run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
