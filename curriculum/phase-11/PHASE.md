# Phase 11 — Vision, image generation, audio and multimodal models

About 85 hours over about 7 weeks, 13 Topics. For a Learner who has finished Phase 10 (transformer block, tiny GPT, CLIP-ready embeddings, sampling). At the end the Learner can trace a prompt through Stable Diffusion, a waveform through Whisper, an image through a vision-language model, and say how each is evaluated.

Every Topic below lists:
- **Learned when** — what the Learner must show, on top of the standard rule (Exercises pass without hints, then later Spaced Reviews).
- **Teach** — the concepts the Tutor draws out through questions. The Tutor never lectures them wholesale.
- **Probe** — misconceptions the Tutor actively tests for during lessons and Spaced Reviews.
- **Sources** — the pages the Tutor teaches against.
- **Exercises** — folder names under this directory, in order.

Exercise folder layout: `exercise.md` (problem, examples, constraints, Hint Ladder, and the questions the Breakdown answers), then `starter.py` / `test.py` / `reference.py`. Exercises are Python-only. They may use numpy and PyTorch on CPU (both installed). Nothing downloads pretrained weights or datasets: every input is tiny and synthetic with fixed seeds, and each test file runs in well under 20 seconds on a laptop CPU. Numeric results are checked with tolerances on tiny tensors. The reference is only used by `scripts/verify-exercises.mjs` to prove the tests are correct; it is shown in the Breakdown once the Learner's own solution passes.

---

## 1. From CNNs to the Vision Transformer (ViT)

**Learned when:** the Learner turns a 224×224×3 image into 196 patch tokens plus a [CLS] token and positions, and explains why ViT needs more data than a CNN.

**Teach:** CNN inductive biases (locality, translation equivariance) as a recap; patchify (16×16 → flatten → linear projection); the [CLS] token or mean pooling; learned 2-D positional embeddings; the same transformer encoder as text; weak inductive bias means large pretraining, then strong transfer; ViT is the vision encoder inside CLIP and VLMs.

**Probe:** "ViT replaced CNNs everywhere" (CNNs still win at small data and on edge devices); "patches = pixels"; assuming ViT understands the whole image meaningfully from layer 1; miscounting tokens when the image size is not a multiple of the patch size.

**Sources:** https://arxiv.org/abs/2010.11929 · https://cs231n.stanford.edu/schedule.html

**Exercises:**
- `01-01-conv2d-valid-recap` — `conv2d_valid(img, kernel)` on a single channel, checked against hand results.
- `01-02-patchify` — `patchify(img, p)` on nested lists (H×W×C) → N×(p·p·C) and `num_patch_tokens(H, W, p, cls)`, with count and ordering tests.
- `01-03-patch-embedding` — `patch_embed(img, p, W, pos, cls_vec)` in numpy returning the (N+1)×d token matrix that enters the encoder.

## 2. Contrastive learning and CLIP

**Learned when:** the Learner computes the symmetric InfoNCE loss on an N×N similarity matrix and does zero-shot classification with text prompts.

**Teach:** two encoders (image ViT/ResNet and a text transformer) projected into a shared space; L2 normalization; the similarity matrix and temperature; symmetric cross-entropy where the diagonal holds the positives and in-batch negatives are free; 400M web image-text pairs; zero-shot classification = embed "a photo of a {label}" and take the argmax cosine; prompt templates and ensembling; CLIP as Stable Diffusion's text encoder and as an evaluation metric.

**Probe:** "CLIP generates images"; "CLIP is trained on labelled classes"; "batch size doesn't matter" (it sets the number of negatives); forgetting to normalize before the dot product.

**Sources:** https://arxiv.org/abs/2103.00020

**Exercises:**
- `02-01-zero-shot-classify` — `zero_shot_classify(img_emb, label_embs)` returning the argmax cosine after normalization.
- `02-02-recall-at-k` — `recall_at_k(sim_matrix, k)` for image→text and text→image retrieval.
- `02-03-clip-loss` — `clip_loss(img_embs, txt_embs, temperature)` in numpy: normalize, similarity matrix, symmetric cross-entropy; low for aligned pairs, high when shuffled.

## 3. Autoencoders, VAEs and VQ-VAE

**Learned when:** the Learner writes the ELBO, applies the reparameterization trick, quantizes vectors against a codebook, and explains why Stable Diffusion's VAE makes diffusion cheap.

**Teach:** an autoencoder compresses then reconstructs through a bottleneck; the VAE encoder outputs μ and σ, with z = μ + σ·ε; loss = reconstruction + KL(q(z|x) ‖ N(0, I)) in closed form; blurry samples; VQ-VAE snaps to the nearest codebook vector to give discrete tokens (the bridge to image and audio tokens and codecs); latent-space interpolation.

**Probe:** "an autoencoder is a generative model" (a plain AE has no prior to sample from); "the KL term is just regularization with no role in sampling"; "latents are compressed JPEGs"; sampling ε from the wrong distribution.

**Sources:** https://lilianweng.github.io/posts/2018-08-12-vae/

**Exercises:**
- `03-01-reparameterize-and-kl` — `reparameterize(mu, log_var, eps)` and closed-form `kl_gaussian(mu, log_var)` that is 0 at μ = 0, log σ² = 0.
- `03-02-vq-quantize` — `vq_quantize(vectors, codebook)` returning indices and quantized vectors, with deterministic nearest-neighbour tie-breaking.
- `03-03-tiny-autoencoder-torch` — an `AE(d_in, d_latent)` module trained on seeded synthetic data lying on a low-dimensional manifold until the reconstruction error falls below a threshold.

## 4. GANs, briefly

**Learned when:** the Learner explains the generator/discriminator minimax game, mode collapse, and why diffusion displaced GANs for text-to-image while GANs survive as vocoders and codec adversaries.

**Teach:** G maps noise to a sample and D classifies real vs fake; the minimax value and the optimal discriminator p_data / (p_data + p_g); the non-saturating generator loss; training instability and mode collapse; fast single-step sampling; adversarial losses inside other systems (HiFi-GAN, EnCodec).

**Probe:** "GANs are obsolete everywhere"; "D is kept after training"; expecting the losses to converge like a supervised model; thinking sample quality implies sample diversity.

**Sources:** https://huggingface.co/learn/audio-course/chapter6/pre-trained_models (HiFi-GAN vocoder) · https://arxiv.org/abs/2210.13438

**Exercises:**
- `04-01-gan-losses` — `d_loss(d_real, d_fake)` and `g_loss_nonsat(d_fake)` from probabilities, with the equilibrium value at D = 0.5.
- `04-02-mode-coverage` — `mode_coverage(samples, centers, radius)` as the fraction of true modes hit, flagging collapse.
- `04-03-minimax-value` — `optimal_d(p_data, p_g)` and `minimax_value(p_data, p_g)` over a discrete support, equal to −ln 4 when p_g = p_data.

## 5. Diffusion models: DDPM to DDIM

**Learned when:** the Learner noises x₀ to any x_t in closed form, states L_simple, and explains why sampling needs many steps and how DDIM cuts them.

**Teach:** the forward process q(x_t | x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I); the closed form x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε; linear and cosine noise schedules; the network (UNet or DiT) predicts ε given (x_t, t); L_simple = ‖ε − ε_θ(x_t, t)‖²; reverse sampling step by step; DDIM as deterministic sampling with fewer steps; the score and flow-matching view used by modern models; timestep embeddings.

**Probe:** "the model removes all the noise in one step"; "diffusion predicts the image directly"; "more steps are always better"; mixing up β_t, α_t and ᾱ_t.

**Sources:** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ · https://huggingface.co/learn/diffusion-course/unit0/1

**Exercises:**
- `05-01-beta-schedule-alpha-bars` — `linear_beta_schedule(T, b0, b1)` and `alpha_bars(betas)`, monotone decreasing with ᾱ_T ≈ 0.
- `05-02-q-sample` — `q_sample(x0, t, eps, alpha_bars)` in closed form, with t = 0 ≈ x₀ and the variance at large t checked over seeded draws.
- `05-03-ddpm-loss-and-ddim-step` — `ddpm_loss(eps, eps_pred)`, `predict_x0(x_t, eps_pred, abar)` and a deterministic `ddim_step(x_t, eps_pred, abar_t, abar_prev)`, round-tripping with the true ε.

## 6. Latent diffusion and text-to-image conditioning

**Learned when:** the Learner traces a prompt through Stable Diffusion (CLIP text encoder → UNet cross-attention in a 4×64×64 VAE latent → VAE decode) and explains classifier-free guidance.

**Teach:** diffusing in the latent space (8× downsample per side, 4 channels) is far cheaper; text conditioning through cross-attention with Q from image features and K/V from text tokens (the 77-token CLIP limit); classifier-free guidance: train with dropped text, then ε = ε_uncond + s·(ε_cond − ε_uncond); negative prompts; img2img from a noised input and inpainting; schedulers; DiT in place of the UNet; ControlNet and LoRA adapters as conditioning add-ons.

**Probe:** "higher guidance is always better" (it oversaturates and cuts diversity); "the model stores training images"; "the prompt is read like an LLM reads it" (a weak CLIP text encoder causes attribute-binding failures); img2img starting from pure noise.

**Sources:** https://huggingface.co/learn/diffusion-course/unit3/1 · https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

**Exercises:**
- `06-01-latent-shape-compression` — `latent_shape(h, w, down, ch)` and `compression_ratio(h, w, down, ch)`.
- `06-02-classifier-free-guidance` — `cfg(eps_uncond, eps_cond, scale)` with scale 1 = cond and scale 0 = uncond, and a negative-prompt variant.
- `06-03-img2img-start` — `img2img_start_step(strength, T)` and `noise_input_latent(latent, step, alpha_bars, eps)` producing the starting point for the reverse process.

## 7. Evaluating image models

**Learned when:** the Learner explains FID as the Fréchet distance between Inception-feature Gaussians, CLIP score, and why human and benchmark evaluation are still needed.

**Teach:** FID (lower is better; fragile in sample count, Inception version and image format); IS and KID; CLIP score for text-image alignment and its bias toward CLIP; CLIP directional similarity for edits; prompt suites (PartiPrompts, DrawBench, GenEval); human pairwise preference and Elo arenas.

**Probe:** "FID is comparable across papers"; "a high CLIP score means a good image"; computing FID from a handful of samples; treating an Elo rating as an absolute quality score.

**Sources:** https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation

**Exercises:**
- `07-01-frechet-distance-1d` — `frechet_distance_1d(mu1, s1, mu2, s2)` for the scalar case, zero for identical Gaussians.
- `07-02-elo-update` — `elo_update(ra, rb, outcome, k)` for pairwise arenas, symmetric and zero-sum.
- `07-03-fid-full` — `fid(feats_a, feats_b)` in numpy with the covariance square root via eigendecomposition, zero for identical feature sets and larger for shifted ones.

## 8. Audio as data

**Learned when:** the Learner goes from waveform → STFT → mel spectrogram → log scale and explains sampling rate, Nyquist and bit depth.

**Teach:** sampling rate (16 kHz for speech, 44.1 kHz CD) and Nyquist = sr/2; bit depth; resampling; waveform in the time domain; DFT and the frequency spectrum; STFT with window and hop giving a spectrogram; the mel scale approximating human pitch perception; log-mel as the standard model input (Whisper); decibels.

**Probe:** "a higher sample rate is always better for ASR"; "a spectrogram contains the phase" (the magnitude drops it, which is why a vocoder is needed to invert); "mel = log frequency"; off-by-one frame counts.

**Sources:** https://huggingface.co/learn/audio-course/chapter1/audio_data · https://huggingface.co/learn/audio-course/chapter0/introduction

**Exercises:**
- `08-01-sine-nyquist-frames` — `sine_wave(freq, sr, dur)`, `nyquist(sr)` and `frames(samples, win, hop)`, with sample and frame counts tested.
- `08-02-dft-magnitude` — a naive O(n²) `dft_magnitude(samples)` whose peak lands on the right bin for a pure 1 kHz tone.
- `08-03-mel-scale-and-filterbank` — `hz_to_mel` / `mel_to_hz` (HTK) round-tripping with 1000 Hz ≈ 1000 mel, and `mel_filterbank(n_fft, sr, n_mels)` with triangular filters whose centres increase.

## 9. Speech recognition and Whisper

**Learned when:** the Learner describes Whisper's pipeline (30 s log-mel → encoder → decoder emitting task, language and timestamp tokens) and computes WER.

**Teach:** CTC encoders (wav2vec2-style) vs seq2seq (Whisper); Whisper as an encoder-decoder transformer trained on 680k hours of weakly supervised multilingual, multitask data; special tokens for language, transcribe vs translate and timestamps; 30-second windows and long-form chunking; sizes from tiny to large; robustness from data diversity; WER = (S + D + I) / N; ASR hallucination on silence and music.

**Probe:** "Whisper streams natively"; "WER can't exceed 100%"; "ASR hallucination is impossible" (the decoder is a language model); collapsing CTC repeats after dropping blanks instead of before.

**Sources:** https://github.com/openai/whisper · https://arxiv.org/abs/2212.04356

**Exercises:**
- `09-01-wer` — `wer(ref, hyp)` via word-level Levenshtein returning (S, D, I, N) and the rate, with insertions pushing it past 100%.
- `09-02-ctc-greedy-decode` — `ctc_greedy_decode(frame_ids, blank)` collapsing repeats then dropping blanks, so "hh-e-ll-ll-o" → "hello".
- `09-03-chunk-audio` — `chunk_audio(n_samples, sr, chunk_s, overlap_s)` returning sample ranges that cover everything with the given overlap.

## 10. Audio tokens and neural codecs

**Learned when:** the Learner explains residual vector quantization and why codec tokens let a language model "speak".

**Teach:** a neural codec = encoder → quantized latent → decoder (EnCodec at 24/48 kHz); RVQ: each codebook quantizes the previous residual, giving several token streams per frame; bitrate = frames/s × codebooks × log2(codebook size); semantic tokens (self-supervised) vs acoustic tokens; adversarial and multiscale-spectrogram losses; flattening the streams into one sequence for an LM.

**Probe:** "audio tokens are like text tokens, one per word"; "more codebooks are free" (they multiply sequence length and bitrate); decoding with codebooks in the wrong order; confusing semantic with acoustic tokens.

**Sources:** https://arxiv.org/abs/2210.13438 · https://huggingface.co/learn/audio-course/chapter6/pre-trained_models

**Exercises:**
- `10-01-codec-bitrate` — `codec_bitrate(frame_rate, n_codebooks, codebook_size)`, with 75 Hz × 8 × 1024 = 6 kbps.
- `10-02-rvq-encode-decode` — `rvq_encode(vec, codebooks)` / `rvq_decode(indices, codebooks)` where the error shrinks as codebooks are added.
- `10-03-token-streams-flatten` — `flatten_streams(indices)` / `unflatten_streams(seq, n_codebooks)` turning per-frame codebook indices into one LM sequence and back.

## 11. Text-to-speech

**Learned when:** the Learner compares the three TTS families (acoustic model → mel → vocoder; end-to-end VAE-flow; LM over codec tokens plus a flow-matching or diffusion decoder) and names how TTS is evaluated.

**Teach:** text normalization and phonemes; SpeechT5 (encoder-decoder → log-mel → HiFi-GAN vocoder, speaker x-vectors); VITS/MMS (conditional VAE, no separate vocoder); Bark (semantic → coarse → fine codec tokens); the modern default: a decoder-only LM over interleaved text and speech tokens plus a conditional flow-matching decoder and vocoder; zero-shot cloning from seconds of prompt audio; duration and prosody; evaluation by MOS, WER of ASR on the output and speaker similarity; deepfake and consent concerns.

**Probe:** "TTS is just reading phonemes aloud"; "a vocoder is optional for mel-based systems"; "voice cloning needs hours of data"; reporting MOS without a confidence interval.

**Sources:** https://huggingface.co/learn/audio-course/chapter6/pre-trained_models · https://arxiv.org/html/2601.12480 · https://arxiv.org/pdf/2505.19931

**Exercises:**
- `11-01-normalize-text` — `normalize_text(s)` expanding numbers and abbreviations ("Dr. 3" → "Doctor three").
- `11-02-length-regulator` — `durations_to_frames(phonemes, durs, hop_s)` expanding each phoneme by its duration, with the total frame count tested.
- `11-03-mos-with-ci` — `mos(ratings)` returning the mean and a 95% confidence interval, narrowing as ratings are added.

## 12. Multimodal models

**Learned when:** the Learner draws LLaVA (vision encoder → projector → LLM tokens) and explains the alternatives: cross-attention fusion and native early-fusion models that emit image or audio tokens.

**Teach:** "image as tokens": ViT/CLIP patch features → MLP projector → inserted into the LLM sequence; visual instruction tuning; the image token budget and resolution tiling; cross-attention fusion (Flamingo-style); early fusion with discrete image and audio tokens in one vocabulary; speech-in/speech-out LMs over codec tokens; video as frames × patches and the token-cost explosion; object hallucination.

**Probe:** "the LLM sees pixels"; "multimodal = calling a separate captioner" (that is a pipeline, not a native VLM); "images cost the same tokens regardless of size"; placing image tokens without a placeholder position in the text.

**Sources:** https://arxiv.org/abs/2304.08485 · https://web.stanford.edu/class/cs224n/ (Lec 17: Multimodality)

**Exercises:**
- `12-01-vlm-token-count` — `vlm_token_count(img_h, img_w, patch, tile, max_tiles)` for tiled high-resolution input.
- `12-02-interleave-image-slots` — `interleave(text_ids, image_slots, image_token_id, n_per_image)` building the input sequence with placeholders, positions tested.
- `12-03-projector` — `project(features, W1, b1, W2, b2)` as a two-layer MLP projector in numpy from vision width to LLM width, with a shape check and a PyTorch equivalence test.

## 13. Evaluating generative models

**Learned when:** the Learner chooses between perplexity, exact match, pass@k, benchmark suites, LLM-as-judge and human preference for a given case, and names each one's failure mode (contamination, judge bias, Goodhart).

**Teach:** intrinsic (perplexity) vs extrinsic (task) evaluation; exact match, token F1, BLEU/ROUGE and their weakness for open generation; pass@k for code with the unbiased estimator; MMLU-style multiple choice; contamination; LLM-as-judge and its position, verbosity and self biases; pairwise arenas with Elo/Bradley–Terry; deterministic evals and regression tests in apps; image (FID, CLIP score) and audio (WER, MOS) metrics as a recap.

**Probe:** "a higher benchmark score is better for my use case"; "BLEU measures quality"; "LLM judges are neutral"; estimating pass@k by taking the first k samples.

**Sources:** https://web.stanford.edu/class/cs224n/ (Lec 11: Benchmarking and evaluation) · https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation

**Exercises:**
- `13-01-exact-match-and-token-f1` — `exact_match(pred, gold)` with normalization and SQuAD-style `token_f1(pred, gold)`.
- `13-02-pass-at-k` — `pass_at_k(n, c, k)` as the unbiased estimator, with the edge cases n − c < k and c = 0.
- `13-03-arena-ratings-and-judge-bias` — `bradley_terry_ratings(matches, iters)` from pairwise wins and `judge_position_swap_agreement(a_first, b_first)` measuring position bias.
