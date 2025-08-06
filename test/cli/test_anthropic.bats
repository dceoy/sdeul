#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-4-sonnet-20250514-v1:0}"
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
