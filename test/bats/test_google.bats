#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GOOGLE_MODEL='gemini-1.5-flash'
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --google-model\"" {
  run poetry run sdeul extract \
    --google-model="${GOOGLE_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
