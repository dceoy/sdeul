#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  aws sts get-caller-identity
  export BEDROCK_MODEL='anthropic.claude-3-5-sonnet-20240620-v1:0'
}

teardown_file() {
  :
}

@test "pass with \"sdeul extract --bedrock-model\"" {
  run poetry run sdeul extract \
    --bedrock-model="${BEDROCK_MODEL}" \
    ./test/data/medication_history.schema.json \
    ./test/data/patient_medication_record.txt
  [[ "${status}" -eq 0 ]]
}
