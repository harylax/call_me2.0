*This project has been created as part of the 42 curriculum by haryandr.*

# Call Me Maybe

## Description

### Presentation

**Call Me Maybe** is a project that introduces to **function calling** using **constrained decoding** in Large Language Models. But first, you may be wondering what exactly **function calling**, **Large Language Models (LLMs)** and **constrained decoding** are?

- **LLM:** is an AI model pre-trained to understand and generate human language
- **function calling:** is a mechanism that makes **LLM** translate a natural language request into a precise function with typed parameters, from a list of available functions with descriptions, parameters types and return type.
- **constrained decoding:** is a way to restrict LLM token selection at each generation step to make the output strictly follow predefined structure or format.
- **token:** is a unit of a LLM vocabulary corresponding to an ID. LLMs don't understand raw text directly: text must first be **encoded** (text → token ID) before being processed and used to generate the next tokens, then **decoded** back (token ID → text) so it can be read by humans.

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

The pipeline displays progress visualization to follow the generation live, and uses a custom implementation of encode/decode (`ft_encode`, `ft_decode`) based on a greedy longest-match strategy, much faster than the LLM's BPE (Byte Pair Encoding) for short text, while falling back to BPE which is more accurate for longer text.

## Instructions

### Installation

This project uses `uv` to manage the Python environment and dependencies.

Make sure `uv` is installed.

Since the model weight, and `transformers` and `torch` packages are heavy, dedicated cache directories are used:

```bash
export HF_HOME=/home/$(USER)/goinfre/hf_cache
export UV_CACHE_DIR=/home/$(USER)/goinfre/uv_cache
```

To create the vitual environment and install all dependencies:

```bash
make install
```

This runs:

```bash
uv sync
```

### Compilation

There is no traditional compilation step since the project is written in Python.

### Execution

The program can be launched using:

```bash
make run
```

By default, this runs:

```bash
uv run python -m src   # with default input and output files
```

You can run the program using your own input files and specified output path by using:

```bash
uv run python -m src --input input_file --functions_definition functions_definition_file --output output_file
```

#### Other Makefile commands

```bash
make debug           # run the program under the Python debugger
make lint            # run flake8 and mypy
make lint-strict     # run flake8 and mypy --strict
make clean           # remove Python caches
make fclean          # clean and remove .venv, hf_cache and uv_cache
```

## Resources

- **Call Me Maybe 42 Subject**: already clearly explains the process of constrained decoding and the function calling mechanism expected in the project.
- **`llm_sdk` module**: provided wrapper with methods used to interact with the LLM (`get_path_to_vocabulary_json`, `encode`, `decode`, `get_logits_from_input_ids`).
- Official documentation for the LLM used in the project: https://huggingface.co/Qwen/Qwen3-0.6B
- Additional documentation on how function calling works: https://huggingface.co/docs/hugs/guides/function-calling
- Additional documentation on constrained decoding: https://www.aidancooper.co.uk/constrained-decoding/

### AI Usage

AI was used to:
- rephrase sentences and improve their clarity in documentation;
- clarify concepts related to LLMs, tokenization, BPE, constrained decoding and function calling;
- help explain the provided llm_sdk package, its Small_LLM_Model class and its methods;
- investigate performance issues and discuss possible optimizations for tokenization and constrained decoding;
- drafting docstrings following PEP 257.

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

Among the surviving tokens the one with the highest probability is selected.

The process is repeated until a complete function name has been generated.

            build_function_calling_prompt
                    |
              ft_encode -> input_ids
                    |
              generated = ''
                    |
        generated in functions_names? <-----------------------------------
        |                           |                                    |
      yes                           no                                   |
        |                           |                                    |
    return generated              get_logits(input_ids) -> logits        |
                                    |                                    |
                                  Are any function_name                  |
                                  beginning with generated?              |
                                  = remaining_suffixes                   |
                                    |                                    |
                                For each token in fn_name_tokens:        |
                            is it a prefix of one of remaining_suffixes? |
                                    |                                    |
                              masked_logits[token]                       |
                              = logits[token] else -inf                  |
                                    |                                    |
                            best_id = argmax(masked_logits)              |
                                    |                                    |
                            generated += ft_decode(best_id)              |
                              input_ids.append(best_id)    ---------------

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
- 


                    build_params_prompt
                            |
                    ft_encode -> input_ids
                            |
            signed_list = signed digits found in user_prompt
                            |
            For each param in function's parameters,
                    what is the type ?
                            |
        -----------------------------------------------------
        |                       |                           |
      string                number/integer              boolean
        |                       |                           |
    closing_char='"'          closing_char="'"          get_logits(input_ids) -> logits
    seen = {}                 signed_list=...           res[param] = logits[true_id] > logits[false_id]
        |                       |                                   |
        ----_constrained_gen ----                                   |
                  |                                                 |
              generated = ''                                        |
                  |                                                 |
    generated ends with closing_char? <-------------------------    |
        |                   |                                  |    |
        yes                 no                                 |    |
        |                   |                                  |    |
        |          get_logits(input_ids) -> logits             |    |
        |                   |                                  |    |
        |          mask logits by parameter's type:            |    |
        |            string  -> string_tokens (penalize seen)  |    |
        |            number  -> start/mid_number_tokens        |    |
        |            integer -> start/mid_integer_tokens       |    |
        |                   |                                  |    |
        |          best_id = argmax(masked_logits)             |    |
        |                   |                                  |    |
        |          best_token = ft_decode(best_id)             |    |
        |          apply sign from signed_list for first digit |    |
        |          input_ids.append(best_id)                   |    |
        |          seen[best_id] += 1 for string               |    |
        |          generated += best_token                     |    |
        |                   |                                  |    |
        |--- yes ----len(generated) > 50?                      |    |
        |                   |                                  |    |
        |                  no ----------------------------------    |
        |                                                           |
    res[param] = float(number) / int(integer) / stripped string     |
        |                                                           |
        -------------------------------------------------------------
                            |
                      return res

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
                                                Vocabulary analysis 
                                                (startup, once)
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

### 4. Step by step visualisation

Generation can take several seconds.

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

### 6. Use of different models

Other models than the required by the subject were used to test the constrained decoding implemention.

| model                      | Accuracy (11 prompts) | valid JSON | speed test |
|----------------------------|-----------------------|------------|------------|
|`Qwen/Qwen3-0.6B` (default) |        90%+ (10/11)   |        ✅  |  ~2min ✅  |
|`HuggingFaceTB/SmolLM2-360M`|        ~55% (6/11)    |        ✅  |  ~2min ✅  |
|`Qwen/Qwen2.5-0.5B`         |        ~55% (6/11)    |        ✅  |  ~2min ✅  |
|`Qwen/Qwen3-1.7B`           |        90%+ (10/11)   |        ✅  |  ~7min ❌  |

## Performance analysis

- **Near-perfect accuracy**: reached 90%+ correct function selection and parameter extraction on prompts (`data/input/function_calling_tests.json`) and functions definitions (`data/input/functions_definition.json`) provided with the subject, using `Qwen/Qwen3-0.6B`.
- **100% valid JSON**: every output is fully JSON-schema-compliant. The implementation builds a `list[dict[str, Any]]` with the correct keys (`prompt`, `name`, `parameters`) and correctly typed value (`str`, `int`, `float`, `bool`). `json.dump` simply turn the result into a valid JSON file.
- **Reasonable speed**: caching the valid token IDs for each category at startup avoids scanning the full vocabulary (~150,000 tokens) at every generation step. Thanks to this optimization, all test prompts provided with the subject are processed well under the 5-minute limit, using `Qwen/Qwen3-0.6B`.
- **Robust error handling**: malformed input is gracefully rejected by pydantic validation, missing files and other file/directory errors are caught via `OSError` and `JSONDecodeError`, and other edge cases (such as `KeyboardInterrupt`) are handled to avoid a crash mid-run.

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

Validation of the implementation was done through:
- **Provided test files**: `data/input/function_calling_tests.json` and `data/input/functions_definition.json`, then manually checking each entry of `function_calling_results.json` against the expected function name and parameter types.
- **Custom prompts**: manually modifying test prompts to test negative numbers and boolean. Ambiguous prompts such as: "Do something with 5 and 3" or "Reverse it" were also added.
- **JSON validity check**: every output file was parsed back with `json.load` to confirm it is syntactically valid and matches the expected JSON schema.
- **Manual comparison of `ft_encode`/`ft_decode` against the native `encode`/`decode`**: testing the same strings through both paths and comparing the decoded result, to confirm the custom tokenizer wrapper stays correct on short and long texts.
- **Error path testing**: running the program with missing input files, malformed JSON, and an invalid path, to confirm errors are caught and reported.

## Example usage

The program can be run with the default input files (`data/input/function_calling_tests.json` and `data/input/functions_definition.json`) using:

```bash
make run
```
or
```bash
uv run python -m src
```

The output will be created by default in `data/output/function_calling_results.json`.

You can run the progran with your own prompts and functions definition and get the result in a custom path using:

```bash
uv run python -m src --input path/to/your/function_calling_prompts.json --functions_definition path/to/your/functions_definition.json --output path/to/your/function_calling_results.json
```

The program processes each prompt from **input file** and selects the appropriate function and parameters from the functions defined in **functions definition file**.

For example, given the prompts:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

the results are:
```json
[
  {
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```



## Bonus Features

- **Support for multiple LLM models**: beyond `Qwen/Qwen3-0.6B` (the default model required by the subject), `HuggingFaceTB/SmolLM2-360M`, `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen3-1.7B` were also tested to benchmark the constrained decoding implementation. Smaller models show noticeably lower accuracy compared to the default model, while `Qwen3-1.7B` reaches similar accuracy but at a much slower generation speed (see benchmark table above).
- **Recoding the tokenizer**: the native `encode`/`decode` methods are wrapped by a custom greedy longest-match tokenizer using a dictionnary of vocabulary created at startup with `get_path_to_vocab_file`. This strategy is faster than the original BPE tokenizer for short texts. Both functions fall back transparently to the model's native tokenizer for longer text (> 20 characters), where BPE remains more efficient.
- **Performance optimizations via startup caching**: the model vocabulary (~150,000 tokens) is scanned once at startup to pre-compute, for each generation category, the set of valid token IDs:
	- `fn_name_tokens`: tokens made only of characters found in the available function names
	- `string_tokens`: tokens valid inside a JSON string value (containing `"` only if it's the closing character)
	- `start_number_tokens` / `mid_number_tokens`: tokens valid as the first / following characters of a number
	- `start_integer_tokens` / `mid_integer_tokens`: tokens valid as the first / following characters of an integer
	- `true_id` / `false_id`: token IDs corresponding to the boolean literals `true` and `false`

	This avoids rescanning the full vocabulary at every decoding step and is a major contributor to keeping generation under the 5-minute limit.

- **Visualization of the generation process**: at every decoding step, the newly generated token is printed to the terminal in real time, giving continuous feedback on progress and intermediate results while the model generates.
- **Public implementation of tokenizer**: public `ft_encode`, `ft_decode` methods were implemented to provide an alternative to the model's native tokenizer. `ft_encode` first checks for an exact vocabulary match, then uses a greedy longest-match strategy for short texts. `ft_decode` directly reconstructs short texts from the inverted vocabulary. For longer inputs, both methods fall back to the model's native BPE tokenizer to preserve compatibility and efficiency.
- **Demonstration of how encoding and decoding integrate with constrained decoding**: a single run illustrates the full loop between tokenization and generation — `ft_encode` turns the contextual prompt into input IDs, the logits returned by the model are masked at each step using the pre-computed token sets, and `ft_decode` turns the selected token IDs back into the final `str` / `int` / `float` / `bool` values used to build the JSON output.