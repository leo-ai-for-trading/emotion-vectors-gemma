# Emotion Vectors With Gemma

This repository is a laptop-scale implementation of emotion-vector extraction,
probing, and steering experiments for open-weight Gemma models.

The project is based on the paper **"Emotion Concepts and their Function in a
Large Language Model"** by Sofroniew et al. The original paper studies Claude
Sonnet 4.5 and shows that large language models can contain internal linear
representations of emotion concepts. These representations can track emotional
contexts, generalize across situations, and causally influence model behavior.

This repository adapts the first part of that methodology to Gemma:

```text
emotional stories
-> Gemma hidden states
-> average activations
-> emotion vectors
-> probe scores for new text
-> optional activation steering during generation
```

This is an interpretability and representation-analysis project. It does not
claim that Gemma has emotions or subjective experience.

---

## Paper References

Primary paper:

- Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S.,
  Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S.,
  Rivoire, K., Fish, K., Olah, C., Lindsey, J. **"Emotion Concepts and their
  Function in a Large Language Model."** Transformer Circuits Thread, 2026.
  <https://transformer-circuits.pub/2026/emotions/index.html>

Related official article:

- Anthropic. **"Emotion concepts and their function in a large language
  model."** 2026.
  <https://www.anthropic.com/research/emotion-concepts-function>

arXiv record:

- **Emotion Concepts and their Function in a Large Language Model.**
  <https://arxiv.org/abs/2604.07729>

---

## What Has Been Implemented

The repository currently implements a smaller, open-weight version of the
paper's emotion-vector workflow.

Implemented components:

1. Load emotional story datasets from JSONL.
2. Generate synthetic emotional stories with Gemma instruction models.
3. Validate generated story datasets for balance and quality.
4. Load local or Hugging Face Gemma models.
5. Extract hidden states from a selected transformer layer.
6. Average hidden states over later tokens in each story.
7. Build one vector per emotion by averaging activations for that emotion.
8. Subtract a global mean activation across all emotions.
9. Optionally compute neutral-text PCA components and project them out.
10. Save vectors to compressed `.npz` files.
11. Score probe texts by projecting hidden states onto emotion vectors.
12. Optionally output token-level probe scores.
13. Steer generation by adding an emotion vector to selected model layers.

The project starts small so it can run on a laptop, but the pipeline is designed
so it can scale by increasing:

- number of emotions
- number of topics
- stories per emotion/topic cell
- model size
- validation rigor
- layer sweeps
- steering evaluations

---

## Repository Structure

```text
emotion-vectors-gemma/
  README.md
  pyproject.toml
  data/
    examples/
      neutral_tiny.jsonl
      probe_texts.jsonl
      stories_tiny.jsonl
    generated/
      raw_8x6x2_gemma3_1b_it.jsonl
      stories_8x6x2_gemma3_1b_it.jsonl
      ...
    seeds/
      emotions_small.txt
      topics_small.txt
  scripts/
    compute_vectors.py
    generate_stories.py
    score_texts.py
    steer_generate.py
    validate_stories.py
  src/
    emotion_vectors_gemma/
      activations.py
      datasets.py
      modeling.py
      steering.py
      vectors.py
  tests/
    test_vectors.py
```

Important security note: Hugging Face token files such as `HF_TOKEN` and
`HF_TOKEN.txt` are local-only secrets and are ignored by `.gitignore`. Do not
commit or share token files.

---

## How The Method Works

### 1. Story Dataset

The core input is a JSONL dataset of short stories. Each record has an emotion,
a topic, and story text.

Example shape:

```json
{
  "emotion": "joy",
  "topic": "a train station",
  "story_index": 0,
  "text": "..."
}
```

The paper used a much larger synthetic dataset. This repository begins with
small example data and a small generated dataset so the full pipeline can be
tested locally.

---

### 2. Hidden-State Extraction

For each story, the code runs Gemma with:

```python
output_hidden_states=True
```

The hidden states are extracted from a selected layer. By default, the layer is
chosen automatically around two-thirds of the way through the model.

Implemented in:

```text
src/emotion_vectors_gemma/activations.py
src/emotion_vectors_gemma/modeling.py
```

For each text, the hidden states are averaged over later tokens. This gives one
activation vector per story.

This is a practical proxy for the residual-stream activations used in the paper.

---

### 3. Emotion Vector Construction

Implemented in:

```text
src/emotion_vectors_gemma/vectors.py
```

For each emotion:

1. Collect all story activations for that emotion.
2. Average them.
3. Compute the global mean across all story activations.
4. Subtract the global mean from the emotion mean.

Conceptually:

```text
emotion_vector = mean(activations_for_emotion) - global_mean
```

This produces one direction per emotion.

The saved vector file includes:

- model ID
- selected layer index
- emotion names
- emotion vectors
- global mean
- neutral PCA components
- metadata such as story counts

Vectors are stored as compressed NumPy `.npz` files.

---

### 4. Neutral PCA Projection

The paper uses controls to remove broad non-emotional directions. This
repository implements a lightweight version using neutral texts.

If neutral texts are provided:

1. Extract activations from neutral examples.
2. Run PCA on those neutral activations.
3. Remove selected neutral principal components from the emotion vectors.

This is controlled by:

```bash
--neutral data/examples/neutral_tiny.jsonl
--neutral-variance-threshold 0.5
```

You can skip this with:

```bash
--skip-neutral-pca
```

---

### 5. Probe Scoring

Once vectors are built, new text can be scored against them.

For each probe text:

1. Extract token hidden states.
2. Normalize emotion vectors.
3. Project hidden states onto those vectors.
4. Average scores across tokens.

Implemented in:

```text
scripts/score_texts.py
src/emotion_vectors_gemma/vectors.py
```

The output is JSONL containing one score per emotion.

---

### 6. Steering

The repository includes an exploratory steering script.

It adds an emotion vector into selected transformer layers during generation.

Implemented in:

```text
scripts/steer_generate.py
src/emotion_vectors_gemma/steering.py
```

This is useful for causal experiments:

```text
If we add the "joy" direction, does output become more joy-like?
If we add the "fear" direction, does output become more fear-like?
```

This is exploratory. It is not a safety evaluation and should not be treated as
evidence that the model has subjective emotions.

---

## Setup

From this directory:

```bash
cd emotion-vectors-gemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e .
```

Dependencies are defined in `pyproject.toml`:

```text
accelerate
huggingface-hub
numpy
pandas
scikit-learn
torch
transformers
```

---

## Hugging Face Access

Gemma models on Hugging Face may be gated. You need to:

1. Accept the Gemma model terms on Hugging Face.
2. Create a Hugging Face access token.
3. Export it before running scripts.

Example:

```bash
export HF_TOKEN="hf_..."
```

Do not commit tokens to the repository.

---

## Reproducing The Current Pipeline

### 1. Compute Tiny Emotion Vectors

This uses the small example dataset:

```bash
PYTHONPATH=src python3 scripts/compute_vectors.py \
  --model-id google/gemma-3-270m \
  --stories data/examples/stories_tiny.jsonl \
  --neutral data/examples/neutral_tiny.jsonl \
  --out outputs/gemma_tiny_vectors.npz \
  --device cpu
```

Expected output:

```text
saved <N> vectors to outputs/gemma_tiny_vectors.npz
model=google/gemma-3-270m device=cpu dtype=torch.float32 layer_index=<layer>
emotions=...
```

---

### 2. Score Probe Texts

```bash
PYTHONPATH=src python3 scripts/score_texts.py \
  --model-id google/gemma-3-270m \
  --vectors outputs/gemma_tiny_vectors.npz \
  --input data/examples/probe_texts.jsonl \
  --output outputs/gemma_probe_scores.jsonl \
  --device cpu
```

Optional token-level scores:

```bash
PYTHONPATH=src python3 scripts/score_texts.py \
  --model-id google/gemma-3-270m \
  --vectors outputs/gemma_tiny_vectors.npz \
  --input data/examples/probe_texts.jsonl \
  --output outputs/gemma_probe_scores_with_tokens.jsonl \
  --device cpu \
  --include-token-scores
```

---

### 3. Generate A Balanced Story Dataset

This script uses an instruction-tuned Gemma model to generate synthetic stories
for each emotion/topic pair.

```bash
PYTHONPATH=src python3 scripts/generate_stories.py \
  --model-id google/gemma-3-1b-it \
  --emotions data/seeds/emotions_small.txt \
  --topics data/seeds/topics_small.txt \
  --out data/generated/stories_8x6x2_gemma3_1b_it.jsonl \
  --raw-out data/generated/raw_8x6x2_gemma3_1b_it.jsonl \
  --n-per-prompt 2 \
  --max-retries 2 \
  --device cpu \
  --max-new-tokens 700 \
  --temperature 0.8
```

The generator tries to enforce:

- balanced emotion/topic coverage
- multiple stories per prompt
- no explicit use of the target emotion word
- minimum story length
- removal of prompt artifacts
- removal of repeated junk output

Raw model outputs can be saved separately with `--raw-out` for debugging.

---

### 4. Validate Generated Stories

```bash
PYTHONPATH=src python3 scripts/validate_stories.py \
  --stories data/generated/stories_8x6x2_gemma3_1b_it.jsonl \
  --emotions data/seeds/emotions_small.txt \
  --topics data/seeds/topics_small.txt \
  --expected-per-cell 2 \
  --min-words 35
```

The validator checks:

- total records
- count per emotion
- missing emotion/topic cells
- underfilled cells
- short records
- records that explicitly contain the target emotion word

It exits with status `1` if the dataset fails validation.

---

### 5. Compute Vectors From Generated Stories

```bash
PYTHONPATH=src python3 scripts/compute_vectors.py \
  --model-id google/gemma-3-1b-it \
  --stories data/generated/stories_8x6x2_gemma3_1b_it.jsonl \
  --neutral data/examples/neutral_tiny.jsonl \
  --out outputs/gemma3_1b_it_vectors.npz \
  --device cpu \
  --batch-size 2
```

For activation extraction and PCA, CPU is recommended on Apple Silicon.

---

### 6. Steer Generation

```bash
PYTHONPATH=src python3 scripts/steer_generate.py \
  --model-id google/gemma-3-270m \
  --vectors outputs/gemma_tiny_vectors.npz \
  --emotion joy \
  --prompt "Write a short scene about a student opening an exam result." \
  --strength 0.05 \
  --device cpu
```

You can choose target layers:

```bash
--layers 8,9,10
```

If omitted, the steering code chooses a default middle-layer range.

---

## Current Data And Outputs

The repository already contains small examples and generated artifacts.

Example inputs:

```text
data/examples/stories_tiny.jsonl
data/examples/neutral_tiny.jsonl
data/examples/probe_texts.jsonl
```

Seed lists:

```text
data/seeds/emotions_small.txt
data/seeds/topics_small.txt
```

Generated datasets:

```text
data/generated/stories_8x6x2_gemma3_1b_it.jsonl
data/generated/raw_8x6x2_gemma3_1b_it.jsonl
```

Saved vector and scoring outputs:

```text
outputs/gemma_tiny_vectors.npz
outputs/gemma_probe_scores.jsonl
outputs/tiny_random_vectors.npz
outputs/tiny_random_probe_scores.jsonl
```

---

## Important Differences From The Paper

This repository is not a full reproduction of the paper.

The original paper:

- used Claude Sonnet 4.5
- studied a much larger set of emotions
- used a much larger synthetic dataset
- performed extensive validation
- included causal and behavioral evaluations
- studied alignment-relevant behaviors such as reward hacking, blackmail, and
  sycophancy

This repository:

- uses open-weight Gemma models
- starts with small local datasets
- focuses on vector extraction, probing, and lightweight steering
- is designed for laptop-scale experimentation
- does not claim to reproduce the paper's full behavioral findings

The first milestone here is methodological:

```text
Can we build, save, load, score, and steer with emotion vectors in Gemma?
```

---

## Known Practical Issues

### 1. Gemma Access

Some Gemma models require Hugging Face authentication. If you see a gated-repo
or 401 error, make sure:

```bash
export HF_TOKEN="hf_..."
```

is set and that you have accepted the model terms.

### 2. Apple Silicon MPS Instability

For hidden-state extraction, Apple Silicon MPS can produce NaN or Inf hidden
states in this setup. The activation code explicitly checks for this and raises
a helpful error.

Recommended:

```bash
--device cpu --dtype float32
```

for vector computation and probe scoring.

### 3. Chat Template Dependency

Some instruction-tuned Gemma tokenizers require `jinja2>=3.1.0` for
`apply_chat_template`.

The story-generation script includes a fallback Gemma chat prompt if that
dependency is old, but upgrading is still recommended:

```bash
python -m pip install -U "jinja2>=3.1.0"
```

### 4. Dataset Scale

The current datasets are small. Small emotion vectors can be noisy and should
not be overinterpreted. Scaling requires more stories per emotion/topic pair and
stronger validation.

---

## Tests

The repository includes a small test suite:

```text
tests/test_vectors.py
```

Run tests with:

```bash
PYTHONPATH=src python3 -m pytest
```

The tests focus on the vector math utilities, including projection and vector
construction behavior.

---

## Recommended Next Steps

To move this from a prototype toward a stronger application:

1. Remove secrets from the project directory.
2. Add `.gitignore` entries for tokens, caches, and large generated outputs.
3. Expand `emotions_small.txt` and `topics_small.txt`.
4. Generate a larger balanced dataset.
5. Validate the generated stories.
6. Compute vectors across multiple layers.
7. Compare probe accuracy across layers.
8. Add a held-out validation set.
9. Run steering experiments with positive and negative controls.
10. Add plots for token-level probe scores.
11. Compare Gemma 270M, 1B, and larger Gemma models.
12. Document which layer and dataset size produce the most reliable vectors.

---

## Summary

This repository implements a practical, open-weight adaptation of the emotion
vector methodology from Sofroniew et al.'s Transformer Circuits paper.

It currently supports:

- synthetic emotional story generation
- story validation
- Gemma hidden-state extraction
- emotion-vector construction
- neutral PCA projection
- probe scoring
- token-level scores
- activation steering
- saved vector artifacts

It is best understood as a starting point for mechanistic interpretability
experiments on emotion-like internal representations in Gemma.
