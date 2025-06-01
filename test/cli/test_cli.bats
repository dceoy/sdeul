#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
}

teardown_file() {
  :
}

@test "pass with \"sdeul --version\"" {
  run poetry run sdeul --version
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul --help\"" {
  run poetry run sdeul --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate\"" {
  run poetry run sdeul validate \
    ./test/data/medication_history.schema.json \
    ./test/data/medication_history.json
  [[ "${status}" -eq 0 ]]
}
