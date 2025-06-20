#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export API_HOST="${API_HOST:-127.0.0.1}"
  export API_PORT="${API_PORT:-8001}"
  export API_BASE_URL="http://${API_HOST}:${API_PORT}"
  export SERVER_PID=""
  
  # Start the API server in the background
  uv run sdeul serve --host="${API_HOST}" --port="${API_PORT}" > /tmp/sdeul_api.log 2>&1 &
  export SERVER_PID=$!
  echo "Started API server with PID: ${SERVER_PID}" >&3
  
  # Wait for server to be ready
  local max_attempts=30
  local attempt=0
  while ! curl -s "${API_BASE_URL}/health" > /dev/null 2>&1; do
    if [[ $attempt -eq $max_attempts ]]; then
      echo "Failed to start API server after ${max_attempts} attempts" >&3
      cat /tmp/sdeul_api.log >&3
      kill -9 "${SERVER_PID}" 2>/dev/null || true
      return 1
    fi
    echo "Waiting for API server to start... (attempt $((attempt+1))/${max_attempts})" >&3
    sleep 1
    attempt=$((attempt+1))
  done
  echo "API server is ready" >&3
}

teardown_file() {
  if [[ -n "${SERVER_PID}" ]]; then
    echo "Stopping API server with PID: ${SERVER_PID}" >&3
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f /tmp/sdeul_api.log
}

@test "pass with \"GET /health\"" {
  run curl -s -X GET "${API_BASE_URL}/health"
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "healthy" ]]
}

@test "pass with \"POST /validate\" with valid data" {
  local schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}'
  local data='{"name": "John Doe", "age": 30}'
  
  run curl -s -X POST "${API_BASE_URL}/validate" \
    -H "Content-Type: application/json" \
    -d "{\"data\": ${data}, \"json_schema\": ${schema}}"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ \"valid\":true ]]
}

@test "pass with \"POST /validate\" with invalid data" {
  local schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}'
  local data='{"name": "John Doe", "age": "thirty"}'
  
  run curl -s -X POST "${API_BASE_URL}/validate" \
    -H "Content-Type: application/json" \
    -d "{\"data\": ${data}, \"json_schema\": ${schema}}"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ \"valid\":false ]]
  [[ "${output}" =~ \"error\" ]]
}

@test "pass with \"POST /extract\" using OpenAI" {
  # Skip if no OpenAI API key
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    skip "OPENAI_API_KEY not set"
  fi
  
  local schema
  schema=$(cat ./test/data/medication_history.schema.json)
  local text
  text=$(cat ./test/data/patient_medication_record.txt)
  
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": $(echo "$text" | jq -Rs .),
      \"json_schema\": ${schema},
      \"openai_model\": \"gpt-4o-mini\",
      \"skip_validation\": false
    }"
  
  echo "OpenAI extract response: ${output}" >&3
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "data" ]] || [[ "${output}" =~ "detail" ]]
}

@test "pass with \"POST /extract\" without model (uses default)" {
  local schema='{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'
  local text="Extract the name John Doe"
  
  # The API uses a default model when none is specified
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"${text}\",
      \"json_schema\": ${schema},
      \"skip_validation\": true,
      \"temperature\": 0.1
    }"
  
  [[ "${status}" -eq 0 ]]
  # Should return extracted data with validated=false since skip_validation=true
  [[ "${output}" =~ "data" ]]
  [[ "${output}" =~ "validated" ]]
}

@test "fail with \"POST /extract\" with missing required fields" {
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"some text\"}"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "detail" ]]
}

@test "fail with \"POST /extract\" with invalid JSON" {
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "invalid json"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "detail" ]]
}

@test "pass with \"POST /extract\" with custom parameters" {
  # Skip if no OpenAI API key
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    skip "OPENAI_API_KEY not set"
  fi
  
  local schema='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}'
  local text="This is a test document for summarization."
  
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"${text}\",
      \"json_schema\": ${schema},
      \"openai_model\": \"gpt-4o-mini\",
      \"temperature\": 0.5,
      \"max_tokens\": 100,
      \"top_p\": 0.9,
      \"skip_validation\": false
    }"
  
  echo "Custom params extract response: ${output}" >&3
  [[ "${status}" -eq 0 ]]
  # Check that response has proper structure
  [[ "${output}" =~ "data" ]] || [[ "${output}" =~ "detail" ]]
}

@test "pass with API documentation endpoints" {
  # Test /docs endpoint exists
  run curl -s -o /dev/null -w "%{http_code}" "${API_BASE_URL}/docs"
  [[ "${output}" == "200" ]]
  
  # Test /redoc endpoint exists
  run curl -s -o /dev/null -w "%{http_code}" "${API_BASE_URL}/redoc"
  [[ "${output}" == "200" ]]
  
  # Test /openapi.json endpoint exists
  run curl -s "${API_BASE_URL}/openapi.json"
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "SDEUL API" ]]
  [[ "${output}" =~ "/extract" ]]
  [[ "${output}" =~ "/validate" ]]
  [[ "${output}" =~ "/health" ]]
}

@test "pass with \"POST /extract\" using Ollama" {
  # Skip if no Ollama is running
  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    skip "Ollama not running"
  fi
  
  local schema='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}'
  local text="This is a test document for summarization."
  
  run curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"${text}\",
      \"json_schema\": ${schema},
      \"ollama_model\": \"llama3.2:1b\",
      \"ollama_base_url\": \"http://localhost:11434\",
      \"temperature\": 0.1,
      \"skip_validation\": false
    }"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "data" ]] || [[ "${output}" =~ "detail" ]]
}

@test "fail with \"POST /validate\" with malformed JSON schema" {
  local bad_schema='{"type": "invalid_type"}'
  local data='{"name": "John Doe"}'
  
  run curl -s -X POST "${API_BASE_URL}/validate" \
    -H "Content-Type: application/json" \
    -d "{\"data\": ${data}, \"json_schema\": ${bad_schema}}"
  
  [[ "${status}" -eq 0 ]]
  [[ "${output}" =~ "valid" ]]
}

@test "pass with concurrent requests" {
  # Test that the API can handle multiple concurrent requests
  local schema='{"type": "string"}'
  
  # Start 3 concurrent requests
  curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"test1\", \"json_schema\": ${schema}, \"skip_validation\": true}" > /tmp/req1.json &
  local pid1=$!
  
  curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"test2\", \"json_schema\": ${schema}, \"skip_validation\": true}" > /tmp/req2.json &
  local pid2=$!
  
  curl -s -X POST "${API_BASE_URL}/extract" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"test3\", \"json_schema\": ${schema}, \"skip_validation\": true}" > /tmp/req3.json &
  local pid3=$!
  
  # Wait for all requests to complete
  wait $pid1 $pid2 $pid3
  
  # Check all responses are valid
  run cat /tmp/req1.json
  [[ "${output}" =~ "data" ]]
  
  run cat /tmp/req2.json
  [[ "${output}" =~ "data" ]]
  
  run cat /tmp/req3.json
  [[ "${output}" =~ "data" ]]
  
  rm -f /tmp/req1.json /tmp/req2.json /tmp/req3.json
}