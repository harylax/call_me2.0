*This project has been created as part of the 42 curriculum by haryandr.*

# Call Me Maybe

## Description

### Presentation

**Call Me Maybe** is a project that introduces to **function calling** in Large Language Models. But first, you may be wondering what exactly are **function calling** and **Large Language Models (LLMs)**?

- **LLM:** is an AI model pre-trained to understand and generate human language
- **function calling:** is a mechanism that makes **LLM** translate a natural language request into a precise function with typed parameters, from a list of available functions with descriptions, parameters types and return type.

### Goal

So the goal is to make the LLM produce, for every user prompt, the name of the function to call with its parameters (with the correct types), so that the final output strictly complies with JSON schema.

To achieve this, the LLM is provided with:
- a list of natural-language prompts,
- and a list of available functions defined by:
	- a name
	- a description
	- typed parameters
	- a typed return

### Overview

The project pipeline relies on two main functions, run once per prompt:

1. **Function selection** (`function_name_from_llm`)

The model is guided, token by token, until it has generated the exact name of one of the known functions.

2. **Parameters extraction** (`params_from_llm`)

For each parameter of the selected function, the model is guided, token by token, until it has produced a value that matches the parameter's declared type (`string`, `number`, `integer`, `boolean`).

Both steps rely on **constrained decoding**:
- Before every token is picked, the raw logits returned by the model are masked so that only tokens which keep the output on a valid path are ever eligible.
- The final JSON object (`prompt`, `name`, `parameters`) is then assembled by plain Python code and serialized with `json.dump`, so the output is guaranteed to be syntactically valid JSON regardless of what the model does.

The pipeline runs with a threaded progress visualization to follow the generation live, and uses a custom implementation of encode/decode (`ft_encode`, `ft_decode`) based on a greedy longest-match strategy, much faster than the LLM's BPE (Byte Pair Encoding) for short text, while falling back to BPE for longer text.

## Instructions

### Installation

...

### Compilation

...

### Execution

...

## Resources

...

### AI Usage

...

## Algorithm explanation

### 1. Vocabulary Analysis at startup

At startup the model’s vocabulary is scanned.

For each category of data needed to be generated a cache of valid tokens is pre-computed from the scanned vocabulary.

These sets of valid token IDs make the subsequent generation process (function selection and parameter extraction) both faster and more reliable:

- `fn_name_tokens`: tokens that contain only characters appearing in the names of the available functions (computed once the function definitions are loaded).
- `string_tokens`: tokens that may appear inside a JSON string value (tokens containing `"` are allowed only if they end with `"`).
- `start_number_tokens` / `start_integer_tokens`: tokens that are legal as the first character of a number or an integer (digits, optional leading sign).
- `mid_number_tokens` / `mid_integer_tokens`: tokens that are legal after a starting digit/sign/decimal point.
- `true_id` / `false_id`: the token IDs that correspond to the literals `true` and `false`.

### 2. Function selection (`function_name_from_llm`)

For each user prompt a context is built. It lists every available function together with its characteristics (name, typed parameters, description and return type).

This context is encoded into input IDs that is given to the model via `get_logits`. The model returns a probability distribution over its entire vocabulary.

#### Constrained decoding:

Tokens that do not belong to the pre-computed set `fn_name_tokens` are discarded. As are tokens that are not a valid prefix of any remaining function name.

Among the surviving tokens the one with the highest probability is selected (with a small bonus proportional to token length).

The process is repeated until a complete function name has been generated.

### 3. Parameters extraction (`params_from_llm`)

Once a function has been selected, another context is created. It describes the function selected, which parameters are still needed to be generated (with their types), and the original user prompt. Constraints are added to encourage precise answers.

#### Constrained decoding

Each parameter is generated in turn under the token constraints:

- **string**:
	- Only `string_tokens` are allowed.
	- A growing logit penalty is applied to already-generated tokens (via a `seen` dictionary) in order to discourage repetition loops.
	- Generation stops when the closing quote `"` is produced.
- **number / integer**:
	- Only digit, sign and decimal-point tokens are permitted.
	- A `'` delimiter is prepended to the input IDs before generation starts, acting as an internal boundary marker.
	- Generation stops as soon as a token ending with `'` is produced.
	- Because the model rarely emits a leading sign, the original user prompt is scanned for signed numeric literals; the detected sign is then applied to the generated absolute value.
- **boolean**: the logits corresponding to the tokens `true` and `false` are compared; the higher one is chosen.

### 4. Overall pipeline

```
                    Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
                                      |
                        get_path_to_vocabulary_json()
                                      |
                          Model vocabulary (JSON file)
                                      |
                    -------------------------------------
                    |                                     |
              vocab {str: int}                  inv_vocab {int: str}
                    |
                    v
                Vocabulary analysis (startup, once)
                    |
      ----------------------------------------------------------------
      |            |              |               |            |     |
 fn_name_tokens string_tokens  number_tokens  integer_tokens  true_id false_id
      |
      v
        For each prompt in function_calling_tests.json
                          |
                          v
              Build contextual prompt
     (functions list + user prompt + instructions)
                          |
                          v
     -----------------------------------------------------
     |                                                     |
     v                                                     v
Function selection                              (once function is known)
(constrained on fn_name_tokens)                  Parameters extraction
     |                                          (constrained per-type:
     v                                       string / number / integer / bool)
 function name                                              |
     |                                                       v
     ----------------------------------------->  {"name": ..., "parameters": ...}
                                                              |
                                                              v
                                            Append to results list
                                                              |
                                              (repeat for next prompt)
                                                              |
                                                              v
                                                     json.dump(results)
                                                              |
                                                              v
                                     data/output/function_calling_results.json
```

## Design Decision

### 1. Never generating JSON syntax

The LLM is never asked to produce the characters `{`, `}`, `:`, `\n`, space or tab.

Every value it generates is a native Python object (`str`, `float`, `int` or `bool`).

The runner then assembles these values into dictionaries of the form `{"prompt": ..., "name": ..., "parameters": ...}`.

Finally, `json.dump` writes the result as a 100 % valid JSON document.

### 2. Pre-computed sets of valid tokens

At startup the entire vocabulary is scanned and several sets of valid token IDs are cached:

- `fn_name_tokens`
- `string_tokens`
- `start_number_tokens` / `mid_number_tokens`
- `start_integer_tokens` / `mid_integer_tokens`
- `true_id` / `false_id`

Having these sets ready makes the masking step during generation faster.

### 3. Contextual prompts

Instead of sending a minimal prompt, the model receives a rich context that includes:

- the full list of available functions with their signatures and descriptions,
- the original user request,
- explicit instructions about spelling, case sensitivity and avoidance of repetition.

Larger, well-structured prompts significantly reduce hallucinations and improve parameter extraction accuracy.

### 4. Threaded progress visualisation

Generation can take several seconds.

A background thread displays an animated progress bar while the main thread compute the logits.

As token are generated, they are printed in real time. This gives the user continuous feedback on progress and intermediate results.

### 5. Custom tokenizer wrapper (`ft_encode` / `ft_decode`)

A thin wrapper around the model’s tokenizer was implemented.

**`ft_encode`**
- First tries an exact lookup in the vocabulary.
- If the text is longer than 20 characters, falls back to the original model’s `encode`.
- Otherwise applies a greedy longest-match strategy: at each position it selects the longest substring present in the vocabulary.

**`ft_decode`**
- If a single token ID is given, returns the corresponding string from the inverted vocabulary.
- If the list contains more than 20 tokens, falls back to the original model’s `decode`.
- Otherwise simply concatenates the strings obtained from the inverted vocabulary.

Both approaches avoid the cost of a full BPE (Byte Pair Encoding) tokenization for simple vocabulary lookups, which brings a significant performance gain.

BPE is the tokenization algorithm used by modern LLMs (Qwen, LLaMA, GPT, ...). It is smarter than the greedy longest-match strategy, but slower when dealing with short texts.

## Performance analysis

...

## Challenges faced

### 1. Strings containing `"`
Generation stops on the first `"` token, to close the JSON string.
So the a quoted string cannot be generated.
**Not Fixed**

### 2. Signed numbers
The model never generates a leading `-`. Negative prompts came out positive.
**Fix**: scan the original prompt for signed numeric literals. Apply the detected sign to the generated value afterward.

### 3. Infinite generation loops
String generation sometimes never hit the closing `"`, looping on the same token.
**Fix**: add a growing penalty on repeated tokens (`seen` dict), plus a hard cutoff at 50 characters.

### 4. Hallucinations with poor context
With a minimal prompt, the model invented extra informations that are not needed to be generated.
**Fix**: give a full context: all function signatures, the user prompt, clear instructions and restrictions.

### 5. Slow tokenization on short texts
Using the full BPE tokenizer for every small generated value was too slow.
**Fix**: added `ft_encode`/`ft_decode`. They do a direct vocabulary lookup for short texts (< 20 chars) and only fall back to BPE for longer ones.

## Testing strategy

...

## Example usage

...
