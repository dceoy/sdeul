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
