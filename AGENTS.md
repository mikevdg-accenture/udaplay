This application is a tutorial in progress. Offer to assist me (the student) with API issues (both REST APIs and Python APIs), obvious mistakes and errors, issues related to Python syntax, but do not offer to implement unimplemented methods.

To run this application, first ensure that a venv is configured:

```
python3 -m venv .venv
source .venv/bin/activate
# and then
python3 main.py
```

You are already in the correct directory and do not need to `cd udaplay`. 

We use Vocareum rather than OpenAI directly. Vocareum is a billing proxy for OpenAI. Any errors about incorrect API keys suggest that calls to `OpenAI()` need to use a method parameter `base_url` set to "https://openai.vocareum.com/v1".

There is a custom `VocariumEmbeddingFunction` to ensure that Vocareum is used for embeddings in `lib/vector_db.py`.

If you do not have access to write a file, tell the user so and stop. Sometimes the user forgets to enable file editing for agents.
