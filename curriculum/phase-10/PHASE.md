# Phase 10 — Transformers and large language models

About 85 hours over about 7 weeks, 14 Topics. For a Learner who has finished Phase 9 (backprop, PyTorch training loops, embeddings, RNNs). At the end the Learner can build a GPT from tokenizer to sampler, and explain how it was pretrained, aligned and served, including why it hallucinates.

Every Topic below lists:
- **Learned when** — what the Learner must show, on top of the standard rule (Exercises pass without hints, then later Spaced Reviews).
- **Teach** — the concepts the Tutor draws out through questions. The Tutor never lectures them wholesale.
- **Probe** — misconceptions the Tutor actively tests for during lessons and Spaced Reviews.
- **Sources** — the pages the Tutor teaches against.
- **Exercises** — folder names under this directory, in order.

Exercise folder layout: `exercise.md` (problem, examples, constraints, Hint Ladder, and the questions the Breakdown answers), then `starter.py` / `test.py` / `reference.py`. Exercises are Python-only. They may use numpy and PyTorch on CPU (both installed). Nothing downloads pretrained weights or datasets: every input is tiny and synthetic with fixed seeds, and each test file runs in well under 20 seconds on a laptop CPU. Numeric results are checked with tolerances on tiny tensors. The reference is only used by `scripts/verify-exercises.mjs` to prove the tests are correct; it is shown in the Breakdown once the Learner's own solution passes.

---

## 1. Tokenization and BPE

**Learned when:** the Learner trains byte-level BPE on a paragraph, encodes and decodes with it, and explains why an LLM miscounts the letters in "strawberry".

**Teach:** characters vs bytes vs subwords; UTF-8 bytes as the base vocab of 256; BPE training = repeatedly merge the most frequent adjacent pair; the merge list is ordered; encode applies merges by rank, decode concatenates bytes; a regex pre-split (GPT-2/GPT-4 style) stops merges crossing word, number and punctuation boundaries; special tokens get IDs after the vocab; the vocab-size trade-off (shorter sequences vs a bigger embedding table); tokens per word differ by language.

**Probe:** "tokens = words"; "the model sees letters"; thinking the tokenizer is learned by gradient descent; assuming any token sequence decodes to valid UTF-8; applying merges greedily left to right instead of by rank.

**Sources:** https://karpathy.ai/zero-to-hero.html (Let's build the GPT Tokenizer) · https://github.com/karpathy/minbpe

**Exercises:**
- `01-01-pair-counts-and-merge` — `get_pair_counts(ids)` and `merge(ids, pair, new_id)`, tested on `[1, 2, 1, 2, 3]`.
- `01-02-train-bpe-roundtrip` — `train_bpe(text, vocab_size)` over UTF-8 bytes returning (merges, vocab), with encode→decode round-tripping non-ASCII and emoji.
- `01-03-encode-by-rank` — `encode(text, merges)` applying merges in rank order, with a test case where rank order and left-to-right order disagree.

## 2. Embeddings revisited: static, contextual and tied

**Learned when:** the Learner explains the token embedding table as lookup = one-hot × matrix, distinguishes static from contextual vectors, and pools hidden states into a sentence embedding.

**Teach:** token ID → row of a learned (vocab × d_model) matrix; the one-hot matmul equivalence; the distributional hypothesis and word2vec as history; cosine similarity; static (input-layer) vs contextual (hidden-state) embeddings; tying the input embedding with the output unembedding; sentence embeddings for retrieval by pooling hidden states, trained contrastively.

**Probe:** reading embedding dimensions as interpretable features; assuming the same token has the same vector everywhere in the network; conflating LLM token embeddings with retrieval embeddings; forgetting the padding mask when mean-pooling.

**Sources:** https://web.stanford.edu/class/cs224n/ · https://jalammar.github.io/illustrated-transformer/

**Exercises:**
- `02-01-embed-and-tied-logits` — `embed(ids, table)` equal to `one_hot(ids) @ table`, and `tied_logits(h, table)` = `h @ table.T`.
- `02-02-nearest-and-analogy` — `nearest(query, table, k)` by cosine and `analogy(a, b, c, table, vocab)` excluding the input words, on a hand-made king/queen/man/woman table.
- `02-03-mean-pool-sentence` — `mean_pool(hidden, mask)` ignoring padded positions and L2-normalizing, so two padded copies of the same sentence give the same vector.

## 3. Attention and self-attention from scratch

**Learned when:** the Learner hand-computes softmax(QKᵀ/√d_k)V for three tokens, explains what Q, K and V each do, and shows why masking must happen before the softmax.

**Teach:** attention as a soft dictionary lookup; query, key and value projections of the same input (self-) or of another sequence (cross-); dot-product scores; scaling by √d_k to keep the softmax from saturating; softmax over keys so each row sums to 1; the weighted sum of values; the causal mask (−inf above the diagonal before the softmax); padding masks; O(n²) cost in sequence length; permutation-equivariance without positional information.

**Probe:** treating attention weights as an explanation of reasoning; zeroing weights after the softmax instead of −inf before it; thinking Q, K and V are different inputs in self-attention; believing attention began with transformers (Bahdanau-style attention in RNN seq2seq came first).

**Sources:** https://arxiv.org/abs/1706.03762 · https://jalammar.github.io/illustrated-transformer/

**Exercises:**
- `03-01-stable-softmax-rows` — row-wise `softmax(xs)` that subtracts the max, tested with `[1000, 1001]` and a row containing −inf.
- `03-02-scaled-dot-product-attention` — `attention(Q, K, V, causal)` on lists of lists, with the causal output for row 0 equal to V[0] and every weight row summing to 1.
- `03-03-permutation-equivariance` — a test harness showing that permuting the input rows permutes the (non-causal) output rows identically, and that adding a causal mask breaks it.

## 4. The transformer block

**Learned when:** the Learner draws a pre-LN decoder block, names every tensor shape, counts its parameters, and states what positional encodings add.

**Teach:** multi-head attention (split d_model into h heads of d_k, attend in parallel, concat, project with W_O); sinusoidal positions, learned absolute positions (GPT-2) and RoPE (rotates Q and K so the dot product depends on relative distance); the residual stream `x + sublayer(x)`; LayerNorm vs RMSNorm; pre-LN vs post-LN; the FFN (d → 4d → d with GELU or SwiGLU) as per-token computation holding most parameters; dropout.

**Probe:** "more heads = more parameters"; thinking positional encoding is concatenated; believing the FFN mixes tokens (only attention does); "LayerNorm normalizes over the batch"; leaving out the output projection when counting parameters.

**Sources:** https://jalammar.github.io/illustrated-transformer/ · https://arxiv.org/abs/2104.09864 · https://arxiv.org/abs/1706.03762

**Exercises:**
- `04-01-positional-encodings` — `sinusoidal_pe(pos, d_model)` with PE(0) = [0, 1, 0, 1, …], and `apply_rope(vec, pos)` on 2-D pairs where dot(rope(q, m), rope(k, n)) depends only on m − n.
- `04-02-pre-ln-block` — `rms_norm(x, eps)`, `multi_head_attention(x, params, n_heads, causal)` and `transformer_block(x, params)` in numpy, matching a PyTorch reference block loaded with the same weights.
- `04-03-count-gpt-params` — `count_params(vocab, d_model, n_layers, n_heads, ff_mult, tied)` landing on GPT-2 small ≈ 124M.

## 5. Encoder, decoder and encoder-decoder

**Learned when:** given a task (classification, chat, translation, ASR) the Learner picks the architecture family and says which mask and objective it uses.

**Teach:** encoder-only (BERT: bidirectional, masked-LM objective, used for classification and embeddings); decoder-only (GPT: causal mask, next-token prediction, the LLM default); encoder-decoder (the original Transformer, T5, Whisper: the encoder reads the input, the decoder cross-attends to it); cross-attention takes Q from the decoder and K/V from the encoder output; the prefix-LM mask as the middle ground.

**Probe:** "BERT can generate text like GPT"; "decoder-only can't translate"; "encoder-decoder is obsolete" (Whisper and much of TTS/ASR use it); getting Q and K/V sources backwards in cross-attention.

**Sources:** https://jalammar.github.io/illustrated-transformer/ · https://huggingface.co/learn/llm-course/chapter1/1

**Exercises:**
- `05-01-make-mask` — `make_mask(n, kind, prefix_len)` for "bidirectional", "causal" and "prefix", tested against exact boolean grids.
- `05-02-mask-tokens-bert` — `mask_tokens(ids, rate, mask_id, rng)` with a seeded RNG, checking the masking ratio and that labels exist only at masked positions.
- `05-03-cross-attention` — `cross_attention(dec_x, enc_out, params)` with Q from the decoder and K/V from the encoder, whose output shape follows the decoder length and whose weights sum to 1 over encoder positions.

## 6. The language-modelling objective and a tiny GPT

**Learned when:** the Learner trains a tiny GPT on synthetic text in PyTorch, explains the loss curve, and knows what loss ≈ ln(vocab) at init means.

**Teach:** the autoregressive factorization p(x) = Π p(x_t | x_<t); shift-by-one targets; cross-entropy = −log p(correct token) and perplexity = exp(loss); a bigram baseline first; teacher forcing trains all positions in parallel thanks to the causal mask; train/val split and overfitting; (B, T) batch × context tensors; AdamW with warmup and cosine decay; the logits → softmax → sample loop at inference.

**Probe:** thinking the model is trained to write whole answers; "low training loss = good model"; believing generation during training is sequential; forgetting the shift so the model learns to copy its input.

**Sources:** https://karpathy.ai/zero-to-hero.html (Let's build GPT) · https://github.com/karpathy/nanoGPT

**Exercises:**
- `06-01-make-xy-and-ce` — `make_xy(ids, block_size, i)` returning shifted input/target windows, and `cross_entropy(logits, target)` via log-sum-exp where uniform logits give ln(V).
- `06-02-bigram-baseline` — `train_bigram(ids, vocab)` with add-one smoothing and `perplexity(model, ids)`, beating uniform on a seeded synthetic sequence.
- `06-03-tiny-gpt-torch` — a `TinyGPT(vocab, block, d_model, n_heads, n_layers)` module with a causal block and `generate(ids, n)`, which starts near ln(V) loss and overfits a seeded repeating sequence within a few hundred steps.

## 7. Scaling and pretraining

**Learned when:** the Learner estimates training FLOPs ≈ 6·N·D, states the Chinchilla rule of thumb (about 20 tokens per parameter), and describes what a pretraining data pipeline does.

**Teach:** scaling laws (loss falls as a power law in parameters, data and compute); compute-optimal scaling grows parameters and tokens together (Chinchilla 70B beat Gopher 280B at equal compute); over-training past Chinchilla to make inference cheaper; the data pipeline (web crawl, dedup, quality filtering, mixing code and math); bf16 mixed precision; data, tensor and pipeline parallelism as concepts; emergent abilities vs smooth metrics; a base model is not an assistant.

**Probe:** "bigger is always better at fixed compute"; "pretraining teaches instruction following"; "the model memorizes the internet"; counting inference FLOPs as 6ND.

**Sources:** https://arxiv.org/abs/2203.15556 · https://web.stanford.edu/class/cs224n/ (Lec 7: Pretraining)

**Exercises:**
- `07-01-flops-and-chinchilla` — `train_flops(n_params, n_tokens)` = 6ND and `chinchilla_tokens(n_params)` ≈ 20N, tested on 70B → 1.4T.
- `07-02-gpu-days` — `gpu_days(flops, peak_flops, mfu, n_gpus)` with sanity checks that MFU and GPU count scale it as expected.
- `07-03-dedup-and-overlap` — `dedup_exact(docs)` by hashing and `ngram_overlap(a, b, n)` for near-duplicate detection.

## 8. Fine-tuning, instruction tuning and LoRA

**Learned when:** the Learner explains the base → SFT pipeline, formats a chat template with a response-only loss mask, and computes how many parameters LoRA trains.

**Teach:** full fine-tuning vs parameter-efficient; LoRA freezes W and learns ΔW = BA with rank r ≪ d, merged at inference so there is no extra latency; SFT on (prompt, response) demonstrations with the loss masked to response tokens; chat templates and role tokens; catastrophic forgetting; prompting and in-context learning as zero-parameter adaptation; RAG vs fine-tuning for new facts.

**Probe:** "fine-tuning is the best way to add new facts" (it raises hallucination); "LoRA is slower at inference"; including the prompt in the SFT loss; thinking the alpha/r scale is optional.

**Sources:** https://arxiv.org/abs/2106.09685 · https://huggingface.co/learn/llm-course/chapter11/1 · https://web.stanford.edu/class/cs224n/ (Lec 9: PEFT)

**Exercises:**
- `08-01-lora-param-count` — `lora_param_count(d_in, d_out, r)` vs full, tested on 4096×4096 with r = 8.
- `08-02-chat-template-and-loss-mask` — `apply_chat_template(messages)` to a string with role tags and `loss_mask(tokens, roles)` that is 1 only on assistant tokens.
- `08-03-lora-forward-merge` — `lora_forward(x, W, A, B, alpha, r)` equal to `x @ (W + (alpha / r) · B @ A)`, with the merge equivalence tested in numpy.

## 9. Reinforcement learning basics

**Learned when:** the Learner defines an MDP, solves a tiny one by value iteration, runs an epsilon-greedy bandit, and explains the policy-gradient idea (raise the probability of actions that led to high return) well enough to read RLHF.

**Teach:** states, actions, rewards, transitions and discount; a policy; return and value; the Bellman equation and value iteration; exploration vs exploitation with multi-armed bandits; policy gradient intuition (REINFORCE: ∇ log π(a|s) weighted by return) and why a baseline lowers variance; reward hacking when the reward is a proxy; where an LLM fits (the policy emits tokens, the reward arrives at the end of the sequence).

**Probe:** confusing reward with value; thinking the agent needs the transition model to learn (it does not for policy gradient); greedy-only agents never finding the best arm; treating a high-variance single-sample gradient as wrong rather than noisy; assuming more reward always means better behaviour.

**Sources:** https://web.stanford.edu/class/cs234/ · https://rail.eecs.berkeley.edu/deeprlcourse/ · https://huggingface.co/learn (Deep RL course)

**Exercises:**
- `09-01-epsilon-greedy-bandit` — an `EpsilonGreedy(n_arms, eps, rng)` agent with running-mean estimates that picks the best arm most often on a seeded bandit.
- `09-02-value-iteration` — `value_iteration(mdp, gamma, tol)` on a tiny gridworld given as dicts, converging to known values and the greedy policy.
- `09-03-policy-gradient-step` — `reinforce_grad(logits, actions, returns, baseline)` for a softmax policy, checked against a numeric gradient of the expected log-prob-weighted return.

## 10. RLHF and preference optimization (DPO)

**Learned when:** the Learner describes SFT → reward model → PPO with a KL penalty, explains why DPO drops the reward model and the sampling loop, and computes both losses on toy log-probs.

**Teach:** InstructGPT (a 1.3B aligned model preferred over the 175B base); pairwise preference data (chosen vs rejected); the Bradley–Terry reward-model loss −log σ(r_c − r_r); the PPO objective reward − β·KL(π‖π_ref); reward hacking; DPO as a classification loss on log-prob ratios against a frozen reference, no sampling during training; RL with verifiable rewards for reasoning models (GRPO-style), where test-time compute is spent on longer chains of thought.

**Probe:** "RLHF adds knowledge" (it shapes behaviour and style); "the reward model is the human"; "the KL term is optional"; "DPO has no reference model"; expecting the DPO loss to be zero when the policy equals the reference (it is ln 2).

**Sources:** https://huggingface.co/blog/rlhf · https://arxiv.org/abs/2305.18290 · https://arxiv.org/abs/2203.02155

**Exercises:**
- `10-01-bradley-terry-loss` — `bradley_terry_loss(r_chosen, r_rejected)` with the symmetry test and ln 2 at equal rewards.
- `10-02-kl-penalized-reward` — `kl_penalized_reward(r, logp_policy, logp_ref, beta)` per sequence, and `kl_estimate(logp_policy, logp_ref)` over a batch.
- `10-03-dpo-loss` — `dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)` from log-probs, equal to ln 2 when the policy equals the reference and decreasing as the chosen margin grows.

## 11. Inference I: decoding and sampling

**Learned when:** the Learner implements temperature, top-k and top-p, predicts what each does to a given distribution, and samples deterministically from a seed.

**Teach:** greedy decoding; beam search (repetitive, still used in translation and ASR); temperature divides logits by T, with T → 0 giving greedy; top-k keeps a fixed pool; top-p (nucleus) keeps the smallest set with cumulative probability ≥ p; repetition and frequency penalties; stop sequences; seeds and determinism; constrained decoding for structured output by masking invalid tokens.

**Probe:** "temperature 0 is fully deterministic in production" (batching and kernels can still vary); "top-p = 0.9 means 90% of tokens"; "temperature changes what the model knows"; renormalizing before filtering instead of after.

**Sources:** https://huggingface.co/blog/how-to-generate · https://github.com/kamranahmedse/developer-roadmap (ai-engineer: sampling parameters)

**Exercises:**
- `11-01-temperature-top-k-top-p` — `apply_temperature(logits, T)`, `top_k_filter(probs, k)` and `top_p_filter(probs, p)` returning renormalized distributions, with tie and exact-p cases.
- `11-02-sample-inverse-cdf` — `sample(probs, rng)` via the inverse CDF with a seeded `random.Random`, tested on empirical frequencies over 10k draws.
- `11-03-constrained-and-penalized` — `constrained_mask(logits, allowed_ids)` so only allowed tokens are ever sampled, and `repetition_penalty(logits, history, penalty)`.

## 12. Inference II: KV cache, quantization and context windows

**Learned when:** the Learner computes KV-cache memory for a model and context length, explains prefill vs decode, and quantizes a weight vector to int8 with a bounded error.

**Teach:** the KV cache stores past K and V per layer so each decode step processes one token; prefill is parallel and compute-bound, decode is sequential and memory-bandwidth-bound; cache bytes = 2 · layers · kv_heads · d_head · T · bytes; MQA/GQA shrink it; quantization from fp32 to bf16, int8 and int4 with absmax or zero-point, per-tensor vs per-group; PTQ vs QAT; outliers (LLM.int8, SmoothQuant, GPTQ, AWQ, GGUF); distillation; mixture-of-experts (active vs total parameters); context-window limits (positional range, O(n²) attention, KV memory); lost-in-the-middle degradation; prompt caching.

**Probe:** "the KV cache stores the queries too"; "quantization always makes compute faster" (it mainly saves memory and bandwidth); "a 1M-token window is used equally well throughout"; "an MoE with 400B parameters costs 400B per token".

**Sources:** https://huggingface.co/blog/kv-cache · https://lilianweng.github.io/posts/2023-01-10-inference-optimization/ · https://huggingface.co/docs/transformers/main/en/quantization/overview

**Exercises:**
- `12-01-kv-cache-and-moe-math` — `kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch, bytes_per)` on a Llama-style config showing the GQA saving, and `moe_active_params(shared, per_expert, n_experts, top_k)`.
- `12-02-quantize-int8-groupwise` — `quantize_absmax_int8(xs)` / `dequantize` with max error ≤ scale/2 and zeros preserved, then `quantize_groupwise(xs, group, bits)` where one group's outlier does not hurt the others.
- `12-03-incremental-attention` — `incremental_attention(cache, q, k, v)` appending to a KV cache and returning the same last row as full causal attention.

## 13. Hallucination, mechanically

**Learned when:** the Learner explains hallucination as fluent sampling from p(next token) with no truth check, and names the training, decoding and data causes with a mitigation for each.

**Teach:** the objective rewards plausibility, not truth; the model must emit some token and has no built-in "unknown" unless trained for it; rare facts are poorly stored; sampling can pick low-probability continuations; exposure bias (an early wrong token conditions everything after); fine-tuning on new knowledge increases hallucination; calibration; detection by sampling consistency and retrieval-backed checking; mitigation by RAG, citations, chain-of-verification, abstention training and factuality rewards.

**Probe:** "hallucination is a bug that will be patched"; "temperature 0 prevents it"; "the model knows it is lying"; "RAG eliminates it"; reading a high sequence probability as evidence of truth.

**Sources:** https://lilianweng.github.io/posts/2024-07-07-hallucination/ · https://arxiv.org/abs/2203.02155

**Exercises:**
- `13-01-consistency-score` — `consistency_score(samples)` as the fraction of sampled answers agreeing with the majority, with a flag below a threshold.
- `13-02-expected-calibration-error` — `expected_calibration_error(confidences, correct, n_bins)` with a perfectly calibrated case giving 0.
- `13-03-greedy-path-prob` — `greedy_path_prob(dists, path)` and `greedy_path(dists)` on a toy table where the most probable fluent sequence is the wrong answer.

## 14. Tokenization revisited and LLM quirks

**Learned when:** the Learner traces a failure (spelling, arithmetic, trailing whitespace, non-English cost, glitch tokens) back to tokenization and budgets a context window in tokens.

**Teach:** digit chunking and arithmetic; leading-space tokens; unequal token cost across languages; glitch and undertrained tokens; context-budget accounting (system prompt, history, reserve for the answer); why the same text costs different tokens across models.

**Probe:** "the model can't count because it's dumb"; "same text = same token count across models"; trimming a prompt by characters and expecting proportional token savings; forgetting to reserve tokens for the output.

**Sources:** https://karpathy.ai/zero-to-hero.html · https://github.com/karpathy/minbpe

**Exercises:**
- `14-01-token-cost` — `token_cost(text, merges)` compared across an English and a non-Latin sample of the same meaning.
- `14-02-split-digits` — `split_digits(text)` as a pre-tokenizer rule and `token_count_delta(text, merges)` showing its effect on the count.
- `14-03-context-budget` — `fits_context(message_token_counts, limit, reserve)` and `trim_history(messages, counts, limit, reserve)` dropping the oldest turns first while keeping the system prompt.
