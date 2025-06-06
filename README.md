sdeul
=====

Structural Data Extractor using LLMs

[![CI/CD](https://github.com/dceoy/sdeul/actions/workflows/ci.yml/badge.svg)](https://github.com/dceoy/sdeul/actions/workflows/ci.yml)

Installation
------------

```sh
$ pip install -U sdeul
```

Usage
-----

### Command Line Interface

1.  Create a JSON Schema file for the output

2.  Prepare a local model GGUF file or model API key.

3.  Extract structural data from given text using `sdeul extract`.

    Example:

    ```sh
    # Use OpenAI API
    $ sdeul extract --openai-model='gpt-4o-mini' \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use Amazon Bedrock API
    $ sdeul extract --bedrock-model='us.anthropic.claude-3-7-sonnet-20250219-v1:0' \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use Groq API
    $ sdeul extract --groq-model='llama-3.3-70b-versatile' \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use Ollama API
    $ sdeul extract --ollama-model='gemma3:27b' \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use a GGUF file
    $ sdeul extract --model-file='google_gemma-3-27b-it-Q4_K_M.gguf' \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt
    ```

    Expected output:

    ```json
    {
      "MedicationHistory": [
        {
          "MedicationName": "Lisinopril",
          "Dosage": "10mg daily",
          "Frequency": "daily",
          "Purpose": "hypertension"
        },
        {
          "MedicationName": "Metformin",
          "Dosage": "500mg twice daily",
          "Frequency": "twice daily",
          "Purpose": "type 2 diabetes"
        },
        {
          "MedicationName": "Atorvastatin",
          "Dosage": "20mg at bedtime",
          "Frequency": "at bedtime",
          "Purpose": "high cholesterol"
        }
      ]
    }
    ```

Run `sdeul --help` for more details.

### REST API

SDEUL also provides a REST API for extracting structured data and validating JSON.

1.  Start the API server:

    ```sh
    # Using the installed package
    $ sdeul-api

    # Or using uvicorn directly
    $ uvicorn sdeul.api:app --reload

    # Or using the run script
    $ python run_api.py
    ```

2.  The API will be available at `http://localhost:8000` with the following endpoints:

    - `POST /extract` - Extract structured data from text
    - `POST /validate` - Validate JSON data against a schema
    - `GET /health` - Health check endpoint
    - `GET /docs` - Interactive API documentation

3.  Example API usage:

    ```sh
    # Extract data using OpenAI
    $ curl -X POST "http://localhost:8000/extract" \
      -H "Content-Type: application/json" \
      -d '{
        "text": "Patient is taking Lisinopril 10mg daily for hypertension.",
        "json_schema": {
          "type": "object",
          "properties": {
            "medications": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "dosage": {"type": "string"},
                  "condition": {"type": "string"}
                }
              }
            }
          }
        },
        "openai_model": "gpt-4o-mini",
        "openai_api_key": "your-api-key"
      }'

    # Validate JSON data
    $ curl -X POST "http://localhost:8000/validate" \
      -H "Content-Type: application/json" \
      -d '{
        "data": {"medications": [{"name": "Lisinopril", "dosage": "10mg", "condition": "hypertension"}]},
        "json_schema": {
          "type": "object",
          "properties": {
            "medications": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "dosage": {"type": "string"},
                  "condition": {"type": "string"}
                }
              }
            }
          }
        }
      }'
    ```
