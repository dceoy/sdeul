#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
}

teardown_file() {
  :
}

@test "pass with \"sdeul --version\"" {
  run uv run sdeul --version
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul --help\"" {
  run uv run sdeul --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul extract --help\"" {
  run uv run sdeul extract --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate --help\"" {
  run uv run sdeul validate --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul serve --help\"" {
  run uv run sdeul serve --help
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate\"" {
  run uv run sdeul validate \
    ./test/data/medication_history.schema.json \
    ./test/data/medication_history.json
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate\" with multiple files" {
  run uv run sdeul validate \
    ./test/data/medication_history.schema.json \
    ./test/data/medication_history.json \
    ./test/data/medication_history.json
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate\" with --info flag" {
  run uv run sdeul validate \
    --info \
    ./test/data/medication_history.schema.json \
    ./test/data/medication_history.json
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"sdeul validate\" with --debug flag" {
  run uv run sdeul validate \
    --debug \
    ./test/data/medication_history.schema.json \
    ./test/data/medication_history.json
  [[ "${status}" -eq 0 ]]
}
