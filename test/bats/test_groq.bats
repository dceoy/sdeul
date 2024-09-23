#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GROQ_MODEL='llama-3.1-70b-versatile'
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --groq-model\"" {
  run poetry run sdeul extract \
    --groq-model="${GROQ_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
