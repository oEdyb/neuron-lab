# Publishing the Ministral map, and scoring neuron labels

Checked 2026-11-03. Research only: nothing was built or uploaded. This is not legal advice.

## Answers

**Q1. Can we publish the Ministral map? Yes, with conditions.** Ministral 3 3B is Apache-2.0 and ungated, and neither the license nor Mistral's published terms prohibit publishing data derived from it. Publish it with a dataset card that carries the Apache license text, says what we made and changed, credits Mistral, and doesn't imply Mistral endorses it.

**Q2. How do we score a label? Run a blind steering test against a random control.** For each Library entry, generate 10 replies on fixed held-out prompts at its saved dose, plus 10 replies with a random direction of the same size. Haiku 5.5 grades each reply 0–2 on concept, instruction and fluency, without knowing which is which. Estimated cost for 100 entries: 30–60 minutes on a Mac plus about $0.13 of judge calls, or under $0.50 in total on a Modal L4. The compute times are estimates, not measurements.

## Q1: Publishing the map

### What the license says

Mistral's [BF16 model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), the weights we load:

> This model is licensed under the Apache 2.0 License.
>
> *You must not use this model in a manner that infringes, misappropriates, or otherwise violates any third party's rights, including intellectual property rights.*

[Mistral's launch post](https://mistral.ai/news/mistral-3) (2 December 2025): "All models are released under the Apache 2.0 license."

The Hugging Face API reports `license: apache-2.0` and `gated: False` for both the BF16 and FP8 repositories. The file tree at our pinned revision `b6d637b` has no LICENSE or NOTICE file, and a request for NOTICE on `main` returns 404.

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt), the parts that matter:

> "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

> 4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions: (a) You must give any other recipients of the Work or Derivative Works a copy of this License; and (b) You must cause any modified files to carry prominent notices stating that You changed the files; and (c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work [...]

> 6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

Mistral's other terms:

- [Usage Policy](https://legal.mistral.ai/terms/usage-policy) (effective 11 June 2026): "This Usage Policy does not apply to Mistral AI Products deployed on a customer's infrastructure, on the infrastructure of our partners, or to our open-source AI models and products."
- [Commercial Terms](https://legal.mistral.ai/terms/commercial-terms-of-service) (effective 25 September 2026): these never mention Apache or open-weight models. They cover "Mistral AI Studio, Vibe, and the other websites, products, software, services, and technologies we offer". They also say model weights are not "Output".
- [License Notice](https://legal.mistral.ai/terms/license-notice) (effective 28 November 2025): this only covers Mistral Code and doesn't address open-weight models.

### What each piece of data needs

| Data | In the map today? | Allowed? | What it needs |
|---|---|---|---|
| KL, value and spread numbers, logit lens, column lengths, map dots, Web connections | Yes. Computed from the weights and our own prompts (`map.py`, `build_view.py`, `connections.py`) | Yes | Apache notice and a statement of how it was made |
| Words per neuron | Yes. Decoded from Mistral's tokenizer, in the same Apache repository | Yes | Same |
| Labels and notes | Library only, not in the published map | Yes, they are our own writing | Mark them as our claims, not Mistral's |
| Generated replies | In our docs | Nothing found against it | Disclose the unfilled `{today}` date in the system prompt |
| Top-activating examples | **No.** The map has no corpus | Depends on the corpus | **The corpus's license decides this, not Mistral's.** Check it, and check for personal data, before adding any |
| Meaning search index | Not built for Ministral | Yes | The vectors come from [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (`build_explorer.py:36`), also Apache-2.0, so credit that model too |

A correction to the question as asked: our map doesn't contain activations in the usual sense. It records how the next word changes when a neuron is pushed. That matters for Q2 as well.

### Our existing maps

[`oExz/neuron-lab-maps`](https://huggingface.co/datasets/oExz/neuron-lab-maps) holds `Qwen3.5-2B` and `gemma-4-E2B-it`, and no Ministral map yet.

The same license covers both models, but **we aren't meeting its conditions cleanly today**. The dataset has no README and no license tag: `README.md` returns "Entry not found", `cardData` is empty, and `publish_maps.py` only uploads data files. The repository's MIT license covers our code, not this dataset. If the maps count as derivative works, they are missing what Section 4 asks for.

| Model | License found | Note |
|---|---|---|
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache-2.0 tag, and a `LICENSE` file with the Apache text | No NOTICE file |
| [Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Apache-2.0 tag, linking to the [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license), which is the Apache text | See the uncertainty below about the older prohibited-use policy |

### Before publishing

1. Add a dataset `README.md` with `license: apache-2.0`, and add the Apache text as `LICENSE`. This fixes all three maps at once.
2. For each map, state the source model, the pinned revision, and that it is derived measurement data with no weights included. For Ministral, also state that it used the BF16 weights, the text decoder only, and the unfilled date placeholder.
3. Credit Mistral AI, Qwen, Google and the MiniLM authors. Say the project is not affiliated with or endorsed by any of them.
4. Warn that the words and replies can include offensive text. The README already warns about slurs at high doses.
5. Have `publish_maps.py` upload the card, so a later publish can't leave it out.

### What's uncertain

- **Is the map a derivative work at all?** It may not be, and whether model weights or measurements of them are copyrightable at all is unsettled. I recommend following Section 4 anyway, since it costs one README and is safe under either reading.
- **Do the Commercial Terms reach a downloaded Apache model?** Their "other ... technologies we offer" wording is broad. I read the specific license on the model card as controlling, but that is my reading, not a statement from Mistral.
- **What is the status of the model card's "must not use" sentence?** It sits beside the license rather than in it. Publishing our own measurements doesn't appear to infringe anyone's rights, so it shouldn't change the outcome.
- **Does Google's older policy still apply?** The [prohibited-use policy](https://ai.google.dev/gemma/prohibited_use_policy) (last modified 21 February 2024) says "Gemma or Model Derivatives" without naming versions, and the site still links it. The Gemma 4 license page has no such terms. Harmless for measurement data, but I couldn't confirm which applies.
- **Not checked:** Hugging Face's content policy for offensive text in datasets, and any statement from Mistral specifically about derived interpretability data. I found none.

## Q2: Scoring labels and steering

### What the methods measure

| Method | Question it answers | Fit here | Cost |
|---|---|---|---|
| Simulate and score ([Bills et al., OpenAI, 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) | Can the label predict when the neuron fires? | Poor: needs a corpus of activations, which we don't have | $3.6k–$219k per 100k features, per Paulo et al. |
| Detection and fuzzing ([Paulo et al., 2024](https://arxiv.org/abs/2410.13928)) | Can a judge use the label to pick out text that activates the neuron? | Poor for now: same missing corpus | $588–$6.2k per 100k features |
| Intervention scoring (Paulo et al.) | Does the label predict what intervening does? | Right question, but its surprisal scorer is harder to run | Not read in detail |
| Output-centric descriptions ([Gur-Arieh et al., ACL 2025](https://arxiv.org/abs/2501.08319)) | Does a description capture the causal effect on output? | Good. Our map's rising words are this kind of data | Low |
| Output score ([Arad et al., 2025](https://arxiv.org/abs/2505.20063)) | Does pushing the feature promote its own logit-lens words? | Good as a filter, but it never reads the label | About 6 s per feature on one GPU |
| **Concept, instruction, fluency judging ([AxBench, Wu et al., 2025](https://arxiv.org/abs/2501.17148))** | **Does steering produce the concept, stay on task, and stay readable?** | **Best: tests the claim the Library actually makes** | **See below** |

What I took from the primary sources:

- **AxBench protocol.** Each generation runs up to 128 tokens and an LLM judge scores concept, instruction and fluency at 0, 1 or 2. The overall score is their harmonic mean, which "heavily penalises poor performance on any of these three subscores". It uses 10 instructions per concept, split in half: one half picks the factor, the other evaluates it.
- **AxBench trade-off.** A stronger push lowers the instruction score for every method. So a Library score only means something at a stated dose.
- **Paulo et al.** Fuzzing correlates best with human scores (Spearman 0.69, against 0.60 for simulation). Embedding similarity is far weaker at 0.32, which is why I wouldn't use cheap embeddings as the main score. They also match intervention strength by KL so that strong pushes don't inflate scores. Our doses differ per entry, so compare scores across entries with care.
- **[Huang et al., 2023](https://arxiv.org/abs/2309.10312).** They tested GPT-4's labels of GPT-2 XL neurons and found "even the most confident explanations have high error rates and little to no causal efficacy". Labels need a causal test.
- **[Tan et al., 2024](https://arxiv.org/abs/2407.12404).** "Steerability is highly variable across different inputs", and some concepts are "brittle to reasonable changes in the prompt". One good demo prompt proves little, so use several.
- **[Language Model Circuits Are Sparse in the Neuron Basis, 2026](https://arxiv.org/abs/2601.22594).** MLP neurons are "as sparse a feature basis as SAEs", which supports scoring single neurons rather than dismissing them.

### The Goodfire essay

[Neural Networks Are Already Clean](https://already-clean.goodfire.com/) (Nick Cammarata, Goodfire; a footnote is dated 30 October 2026) calls itself a working hypothesis. It estimates that in Qwen3-32B "roughly 90% of neuron poles have one main job", from preliminary audits.

Three points apply to us:

- **It warns against simulation scores.** It says 2023 metrics rewarded "explanations that could simulate activations rather than ones that were right, which often come apart".
- **It checks a label several ways:** surrounding context, steering, upstream, downstream, and the negative pole. "The neuron's negative side matters about as much as its positive."
- **It compares against random.** In its Qwen circuit tests, "'Random' removes as many neurons from the same layers."
- **It has a held-out test that needs no corpus.** For 200 random neurons, "an agent read only each neuron's explanation and wrote five new examples that should drive it and five near misses that shouldn't, before we measured any of them." Among the 183 explanations of a positive side, new examples fired strongly 76% of the time and near misses 11%. The explanation won for 164 neurons, tied for 15 and lost for 4. This is Goodfire's own unreviewed result, and the promised technical paper isn't out.

**Public reaction** is split. The [LessWrong linkpost](https://www.lesswrong.com/posts/x4kvabkF2Jk8GLZp8/linkpost-neural-networks-are-already-clean) has roughly six relevant voices. Alex Gibson notes "the technical paper hasn't been released", johnswentworth wants per-neuron evidence, and StefanHex worries max-activating examples "tell a plausible but incomplete story". [One news summary](https://traictory.com/news/2026-10-17-neural-networks-already-clean-interpretability) reports Neel Nanda saying the essay defines monosemantic too loosely, and Leo Gao saying an explanation that can't simulate activations hasn't been demonstrated. I didn't find their original posts, so treat those two as secondhand.

This disagreement bears on our score: **a steering score shows what pushing a neuron does, not what the neuron reads.** The Library should say "steering verified", never "neuron understood".

**Niche find:** [Leverage Is Not Reach](https://arxiv.org/abs/2606.19831) (Hongliang Liu, June 2026) is the closest paper to this project: single MLP neurons, with dose normalized against stream size much like ours. It says gradient attribution anti-predicts control, and gradient slope is what `find.py` ranks by, and it predicts a neuron's collapse ceiling from the weights and one forward pass. It is a single-author preprint with a small test (open/closed verdict correct on 11 of 15 neurons, against a 10 of 15 baseline), and I read the abstract only. Worth a separate look, but not evidence enough to act on yet.

### Recommendation: a blind steering score

**Why this one:** the Library claims "this neuron at this dose makes the model act like X". This is the cheapest method that tests that claim directly. It needs no corpus, it reuses code we have, and a judge call costs a fraction of a cent.

**Per Library entry, at its saved dose and sign:**

1. Generate replies to **10 fixed prompts**, up to 128 tokens, through the normal chat path. Write them once, keep them frozen, and keep them apart from `config.PROMPTS`, `FRAMES` and `HELD_OUT`.
2. Generate the same 10 with a **random direction at the same dose and layer**, as `doses.py` already does with `push_vector`. Also generate unsteered replies once per model, which catches a label the model would meet anyway.
3. Give a judge one reply at a time, in shuffled order, with the label and prompt but no dose, arm or neuron. It returns concept, instruction and fluency at 0–2.
4. Compute the harmonic mean per reply. Report the steered mean, the control mean, and the paired difference with a bootstrap interval.

Replies that `repeating()` flags or `distinct4()` scores low can get fluency 0 without a judge call. For combos, use one random direction per push.

**Run a calibration first.** Score the 12 entries in `examples/library.json` once with true labels and once with labels swapped between entries. If swapped labels don't score clearly lower, the judge or the prompts are wrong and the score shouldn't ship. This costs under a dollar. Also hand-check about 20 judge grades, because I found no evidence on how reliable Haiku 5.5 is at this.

### Cost

Assumptions: 20 replies per entry, about 500 input and 30 output tokens per judgment, and generation of 10–30 steps per second on an L4 or 3.5–7 on a Mac, for a batch of 20. **The generation speeds are my estimates and have not been measured.** The README's one Mac figure (2.3 s for a batch on 0.8B) doesn't state a token count, so it can't anchor this.

| | 12 entries (current seeds) | 100 entries |
|---|---|---|
| Mac time | 5–10 min, estimated | 30–60 min, estimated |
| Modal L4 at [$0.80/hour](https://modal.com/pricing) | Under $0.10 including model load | $0.10–$0.40 |
| Judge: Haiku 5.5 at [$0.10 in and $0.50 out per MTok](https://platform.claude.com/docs/en/about-claude/pricing) | About $0.02 | About $0.13, or half with the Batch API |

Other limits on these numbers:

- **Ministral has no L4 timing.** [Its only benchmarks are on an H100](../ministral3-3b-first-map.md), at about $3.95/hour. If it needs that card, 15 minutes costs about $1.
- **New dependency.** The project has no judge today, so this needs an Anthropic API key. Using `claude -p` would draw on Olle's usage instead.

Run the first 12 entries on the Mac and record the real speed before trusting the 100-entry figures.

### What the Library would show

```
Evil, L17 N5244 +0.6
Steering verified  1.4 / 2     random control 0.2     difference +1.2 (0.8 to 1.5)
concept 1.8   on-task 1.1   fluent 1.5     10 prompts, Haiku 5.5, 2026-11-03
```

*(Illustrative values, not results.)*

- **Three badges plus one empty state:** Verified, Weak, Failed, or Not scored. Starting rule: Verified needs an overall of at least 1.0 and a difference interval above zero. These thresholds are guesses until calibration has run.
- **Keep the subscores visible.** A neuron with concept 2 and fluency 0 has a very different problem from one with concept 0 and fluency 2. A case like "Lord at −0.8", which reads as lordly but breaks down, should show that directly.
- **Scores attach to label and dose.** Editing either marks the entry "Not scored". This matches the Library's rule that each neuron and exact dose is its own entry. The entry would need a new field, because `library.py:39` only allows a fixed set of keys.
- **Keep the existing checks separate.** Bosse recall and `3x + 7 = 22` stay as yes/no coherence checks, since the steering score doesn't test memory or reasoning.

### What this won't tell us

- It doesn't show what the neuron reads or when it fires. **The essay's five-examples, five-near-misses test is the best next step**, because it needs no corpus: about 10 short sentences and one forward pass per entry, plus one cheap call to write them. I didn't pick it first because Library labels describe what pushing does, and a neuron's write effect may not match what makes it fire. The code also has no hook yet for reading a single neuron's activation, and the essay's "fired strongly" needs a maximum activation we haven't recorded. Comparing the two sets of sentences directly avoids that.
- Ten prompts give a wide interval, so small differences mean nothing.
- It doesn't cover the negative pole unless that sign and dose have an entry of their own.
- It doesn't catch side effects outside the label. The Ministral results show why that matters: [a pirate neuron switched the conversation to French](../ministral3-3b-first-map.md).
- It can't score the 240,000 unlabelled neurons, since it needs a label. Arad's output score is the better fit for filtering at that scale, and a rough version might be computable from saved map arrays without a model. I haven't tested that.

## What I checked and couldn't verify

**Read directly:** the Hugging Face API and cards for all three models and our dataset, the file tree at our pinned Ministral revision, the Apache text, Mistral's launch post and three legal pages, the Gemma 4 license and prohibited-use pages, Qwen's LICENSE opening, Modal and Anthropic pricing, AxBench and Paulo et al. full text, and Arad et al. sections on scoring. For the essay, I searched its full text for specific passages but didn't read all of it. In this repository: `README.md`, `doses.py`, `lab.py`, `library.py`, `build_view.py`, `build_explorer.py`, `publish_maps.py`, `map_data.py`.

**Abstracts only:** Gur-Arieh et al., Huang et al., Tan et al., Liu, arXiv 2601.22594.

**Couldn't verify:**

- **Bills et al.'s body text.** The page renders with JavaScript and returned only the outline, and OpenAI's post returned 403. I confirmed the title, authors and date, but its method and numbers here come through Paulo et al. and the Goodfire essay, which cites about 1.7% of neurons as interpretable.
- **AxBench's judge model and exact prompts.** The prompts are in its Appendix J.3, which I didn't read.
- **Whether any of this has been validated for single-neuron persona steering.** AxBench's concepts are SAE-style and not personality traits. Calibration has to establish this.
- **Any real timing or judge accuracy.** No model or judge was run for this document.
