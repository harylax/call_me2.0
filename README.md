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

...

## Instructions

### Installation

### Compilation

### Execution

## Resources

### AI Usage

## Algorithm explanation

### Vocabulary Analysis at startup

At startup the model’s vocabulary is scanned and the tokens that are valid for the different categories that must be generated are pre-computed and cached. These sets of valid token IDs make the subsequent generation process (function selection and parameter extraction) both faster and more reliable:

- `fn_name_tokens`: tokens that contain only characters appearing in the names of the available functions (computed once the function definitions are loaded).
- `string_tokens`: tokens that may appear inside a JSON string value (tokens containing `"` are allowed only if they end with `"`).
- `start_number_tokens` / `start_integer_tokens`: tokens that are legal as the first character of a number or an integer (digits, optional leading sign).
- `mid_number_tokens` / `mid_integer_tokens`: tokens that are legal after a starting digit/sign/decimal point.
- `true_id` / `false_id`: the token IDs that correspond to the literals `true` and `false`.

### Function selection (`function_name_from_llm`)

For each user prompt a context is built that lists every available function together with its characteristics (name, typed parameters, description and return type).  
This context is encoded into a sequence of input IDs that is fed to the model via `get_logits`. The model returns a probability distribution over its entire vocabulary.

Tokens that do not belong to the pre-computed set `fn_name_tokens` are discarded, as are tokens that are not a valid prefix of any remaining function name. Among the surviving tokens the one with the highest probability is selected (with a small bonus proportional to token length). The process is repeated until a complete function name has been generated, constantly checking the remaining suffixes of the candidate functions.

### Parameters extraction (`params_from_llm`)

A context prompt is constructed that describes the function that has just been selected, the list of parameters that still need to be generated (with their types), and the original user prompt. Constraints are added to encourage precise answers.

Then, for each parameter in turn, a short sub-prompt is appended that names the parameter, indicates its rank and restates its type. Generation proceeds under the appropriate token constraints:

- **string**: only `string_tokens` are allowed. A growing logit penalty is applied to already-generated tokens (via a `seen` dictionary) in order to discourage repetition loops. Generation stops when the closing quote `"` is produced.
- **number / integer**: only digit, sign and decimal-point tokens are permitted. Because the model rarely emits a leading sign, the original user prompt is scanned for signed numeric literals; the detected sign is then applied to the generated absolute value. Generation stops when a non-numeric character (or the closing `'`) appears.
- **boolean**: the logits corresponding to the tokens `true` and `false` are compared; the higher one is chosen.

### Never generating JSON syntax

The LLM is never asked to produce the structural characters `{`, `}`, `:`, `\n`, space or tab.  
Every value it generates is a native Python object (`str`, `float`, `int` or `bool`).  
The runner then assembles these values into dictionaries of the form  
`{"prompt": ..., "name": ..., "parameters": ...}`  
and a final `json.dump` produces a 100 % valid JSON document.

## Design Decision

## Performance analysis

## Challenge faced

- Strings that contain the character `"`
- Signed numbers/integers
- Infinite generation loops on string parameters
- Model hallucinations when no context is provided

## Testing strategy

## Example usage
