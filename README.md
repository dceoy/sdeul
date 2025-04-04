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
