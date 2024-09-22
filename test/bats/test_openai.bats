#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export OPENAI_MODEL='gpt-4o-mini'
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --openai-model\"" {
  run poetry run sdeul extract \
    --pretty-json \
    --openai-model="${OPENAI_MODEL}" \
    --openai-api-key="${OPENAI_API_KEY}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}