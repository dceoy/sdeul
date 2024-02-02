sdeul
=====

Structural Data Extractor using LLMs

[![Test](https://github.com/dceoy/sdeul/actions/workflows/test.yml/badge.svg)](https://github.com/dceoy/sdeul/actions/workflows/test.yml)
[![CI to Docker Hub](https://github.com/dceoy/sdeul/actions/workflows/docker-compose-build-and-push.yml/badge.svg)](https://github.com/dceoy/sdeul/actions/workflows/docker-compose-build-and-push.yml)
[![Release on PyPI and GitHub](https://github.com/dceoy/sdeul/actions/workflows/python-package-release-on-pypi-and-github.yml/badge.svg)](https://github.com/dceoy/sdeul/actions/workflows/python-package-release-on-pypi-and-github.yml)

Installation
------------

```sh
$ pip install -U sdeul
```

Usage
-----

1.  Prepare a Llama 2 GGUF file.

    Example:

    ```sh
    $ curl -SLO https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
    ```

2.  Create a JSON Schema file for the output

3.  Extract structural data from given text using `sdeul extract`.

    Example:

    ```sh
    sdeul extract --pretty-json \
        llama-2-13b-chat.Q4_K_M.gguf \
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
