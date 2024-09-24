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

    Example:

    ```sh
    # Set an OpenAI API key
    $ export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    # Set a Groq API key
    $ export GROQ_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    # Download a model GGUF file from Hugging Face
    $ curl -SLO https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
    ```

3.  Extract structural data from given text using `sdeul extract`.

    Example:

    ```sh
    # Use OpenAI API
    $ sdeul extract \
        --openai-model=gpt-4o-mini \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use Groq API
    $ sdeul extract \
        --groq-model=llama-3.1-70b-versatile \
        test/data/medication_history.schema.json \
        test/data/patient_medication_record.txt

    # Use local LLM
    $ sdeul extract \
        --model-file=Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf \
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
