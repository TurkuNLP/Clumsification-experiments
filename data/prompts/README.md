> This document has been co-created, refactored, and cleaned using GPT 5.6.

# Prompt specifications

This directory is the canonical home for version-controlled prompt text. Prompt
files are UTF-8 JSON and are loaded through `clumsification_code.prompts` so
their paths do not depend on the process working directory.

Each externally sourced prompt or rubric carries a short source citation in
the same file as its wording.

The initial schema describes an ordered chat prompt:

```json
{
  "schema_version": 1,
  "id": "method.prompt_name",
  "version": "1",
  "description": "A short human-readable purpose.",
  "required_variables": ["candidate_text"],
  "messages": [
    {
      "role": "user",
      "content": "Evaluate this text:\n{candidate_text}"
    }
  ],
  "metadata": {
    "method": "method_name"
  }
}
```

`schema_version` describes the file format. `version` describes the prompt
wording and should be changed deliberately when that wording changes.

Every template field must be listed exactly once in `required_variables`.
Fields use simple `{name}` placeholders; attribute access, indexing, format
specifiers, and conversions are rejected. Literal braces must use JSON strings
containing `{{` and `}}`, which render as single braces.

Model configuration, inference settings, output parsing, and score transforms
do not belong in prompt files. They remain explicit code or command-line
configuration.
