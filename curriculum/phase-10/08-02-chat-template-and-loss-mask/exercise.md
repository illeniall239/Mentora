# Chat template and response-only loss mask

Topic: 8. Fine-tuning, instruction tuning and LoRA
Difficulty: 2 of 3

## Problem

Supervised fine-tuning turns a list of chat messages into one string with role tags, and then trains the model **only on the assistant's tokens**. Write both steps. Pure Python only.

- `apply_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str` renders each message `{"role": ..., "content": ...}` as

  ```
  <|role|>\n{content}<|end|>\n
  ```

  and concatenates them in order. `role` must be `"system"`, `"user"` or `"assistant"`; raise `ValueError` for any other role or if `messages` is empty. When `add_generation_prompt` is true, append `<|assistant|>\n` at the end so the string is ready for the model to continue; the training data never uses it.

- `loss_mask(tokens: list[str], roles: list[str]) -> list[int]` takes the tokenised template (one string per token) and, for each token, the role of the turn it belongs to. Return a list of `0`/`1` of the same length: `1` for every token of an assistant turn **except** its opening tag `<|assistant|>` (the model is prompted with that, it does not produce it), and `0` for everything else: system and user turns including their tags, and the assistant's opening tag. The assistant's closing `<|end|>` gets `1`: the model must learn to stop. Raise `ValueError` if the lists differ in length or any role is unknown.

## Examples

```
apply_chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])
  → "<|user|>\nHi<|end|>\n<|assistant|>\nHello!<|end|>\n"

apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
  → "<|user|>\nHi<|end|>\n<|assistant|>\n"

tokens = ["<|user|>", "Hi", "<|end|>", "<|assistant|>", "Hello", "!", "<|end|>"]
roles  = ["user",     "user", "user",  "assistant",    "assistant", "assistant", "assistant"]
loss_mask(tokens, roles)   → [0, 0, 0, 0, 1, 1, 1]
```

## Constraints

- At most 64 messages and 4 096 tokens.
- Pure Python only.

## Hints

1. Write the rendered string for a two-message chat by hand, character by character, including every `\n`. Where does each newline go?
2. What is the difference between the string a training example needs and the string an inference request needs? Which one ends in an open assistant tag?
3. For a token in an assistant turn, what single extra check decides whether the model should be graded on it?
4. If you masked the assistant's `<|end|>` to `0`, what would the fine-tuned model never learn to do?

## Explain-back

- Why is the loss masked to response tokens? What would the model spend capacity learning if user turns were scored too, and why is that not what you want at deployment?
- The base model was pretrained on next-token prediction over web text. What does SFT change about it: its knowledge, its format, or both?
- Fine-tuning on a few thousand dialogues can make a model forget skills it had. What is that called, and what does the loss mask have to do with limiting it?
- Prompting with three examples in the context also changes the model's behaviour, with zero parameters updated. When would you choose that over SFT?
