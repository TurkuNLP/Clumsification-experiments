# English fluency evaluation suite: datasets, labels, annotation criteria, agreement, and text profile

**Status:** source-audited documentation for the suite implemented by `run_standard_benchmark_suite`  
**Audit date:** 2 September 2026  
**Scope:** 42 evaluation dimensions: 31 selected NLG-Eval dimensions, 8 dimensions from four standalone datasets, and 3 pairwise diagnostics

This is the **English** track only. The runner can additionally execute Basque, Spanish, and Norwegian checks when `include_multilingual=True`; those are deliberately outside this document and the 42-dimension count.

## How to read this document

This document distinguishes three things that are easy to conflate:

1. **The source dataset's labels.** These are the dimensions its authors annotated.
2. **The dimension actually used by this project.** One source dataset can therefore contribute several separate evals.
3. **This project's four-category fluency interpretation.** The mapping to **grammaticality (G)**, **coherence (C)**, **clarity (Cl)**, and **naturalness (N)** is a project research decision, not a claim made by the source authors.

The canonical inclusion and mapping decisions are in the [benchmark registry](../clumsification_code/evals/benchmark_registry.py), the standalone normalization is in [standalone_benchmarks.py](../clumsification_code/evals/standalone_benchmarks.py), and pairwise diagnostics are assembled in [benchmark_runner.py](../clumsification_code/evals/benchmark_runner.py). The project's broad definition is that fluency concerns grammaticality, coherence, clarity, and naturalness ([project source](../clumsification_code/perturbations/llm_sampled.py)). Operationally:

- **G — grammaticality:** well-formed grammar, morphology, syntax, spelling, and punctuation.
- **C — coherence:** sentences and ideas form a logically consistent, connected whole with topic continuity.
- **Cl — clarity:** the text is independently understandable and not confusing or ambiguous.
- **N — naturalness:** idiomatic, human-like English with appropriate word choice and rhythm.

These glosses synthesize the project's own evaluation rubric ([rubric JSON](../data/prompts/evaluation/rubrics/geval_no_reference.json)); they do not overwrite any source dataset's instructions.

## Evidence and measurement policy

- Dataset claims below use only the checked-in dataset files, authors' repositories, official dataset releases, or the original publications. Secondary surveys and dataset cards written by third parties were not used as evidence.
- For the NLG-Eval portion, the checked-in [NLG-Eval JSONL](../data/benchmarks/NLG-Eval.jsonl) and [metadata CSV](../data/benchmarks/NLG-Eval_meta_info.csv) are the immediate data sources. NLG-Eval itself was assembled for [Themis](https://aclanthology.org/2024.emnlp-main.891/) from 58 earlier human-evaluation datasets. The original publication/repository is still cited for every constituent collection.
- **Lengths are audit statistics, not publication statistics**, unless explicitly identified otherwise. They were recomputed from the exact text passed to the scorer, using whitespace-separated words after trimming. Sentence counts were not used because punctuation and tokenization conventions vary sharply across collections. Means are rounded to one decimal word.
- `n` is the number of valid text/label records in the current checked-in data after the suite loader's filters. Reusing the same text under two labels produces two eval records but does not create a second distinct text.
- “Not reported” means that no numerical inter-annotator agreement (IAA) was found in the original paper or authors' release for the relevant annotations. It does **not** mean that annotators necessarily disagreed or that no quality control occurred.
- Agreement coefficients are not interchangeable. This document names the statistic, unit, and rater pool whenever the publication permits it. System-ranking agreement, item-level rating agreement, ICC, Cohen's kappa, and exact agreement answer different questions.

## Complete eval inventory

In the “all source labels” column, **bold** labels are used by at least one eval in this suite. The final column gives this project's mapping.

| # | Eval name in code | Source collection / task | All source labels | Relevant source label | Project category |
|---:|---|---|---|---|---|
| 1 | `chiang_cohesiveness` | Chiang LLM Evaluation / story generation | **Cohesiveness**, **Grammaticality**, Likability, Relevance | Cohesiveness | C |
| 2 | `chiang_grammaticality` | Chiang LLM Evaluation / story generation | **Cohesiveness**, **Grammaticality**, Likability, Relevance | Grammaticality | G |
| 3 | `coeval_grammaticality` | CoEval / story generation | Character Development, **Clarity**, **Coherence**, Engagement, **Grammaticality**, Length, Relevance | Grammaticality | G |
| 4 | `coeval_coherence` | CoEval / story generation | Character Development, **Clarity**, **Coherence**, Engagement, **Grammaticality**, Length, Relevance | Coherence | C |
| 5 | `coeval_clarity` | CoEval / story generation | Character Development, **Clarity**, **Coherence**, Engagement, **Grammaticality**, Length, Relevance | Clarity | Cl |
| 6 | `hanna_coherence` | HANNA / story generation | **Coherence**, Complexity, Empathy, Engagement, Relevance, Surprise | Coherence | C |
| 7 | `nextchapter_fluency` | The Next Chapter / story generation | **Coherence**, **Fluency**, Interestingness, Logicality, Relatedness | Fluency | G |
| 8 | `nextchapter_coherence` | The Next Chapter / story generation | **Coherence**, **Fluency**, Interestingness, Logicality, Relatedness | Coherence | C |
| 9 | `pplm_fluency` | PPLM / controlled generation | **Fluency** | Fluency | G + Cl |
| 10 | `e2e_naturalness` | E2E NLG / data-to-text | **Naturalness**, Overall Quality | Naturalness | N |
| 11 | `inlg16_naturalness` | INLG 2016 / data-to-text | Informativeness, **Naturalness**, Phrasing | Naturalness | N |
| 12 | `rankme_naturalness` | RankME / data-to-text | Informativeness, **Naturalness**, Overall Quality | Naturalness | N |
| 13 | `webnlg2017_fluency` | WebNLG 2017 / data-to-text | **Fluency**, **Grammaticality**, Semantic Adequacy | Fluency | N |
| 14 | `webnlg2017_grammaticality` | WebNLG 2017 / data-to-text | **Fluency**, **Grammaticality**, Semantic Adequacy | Grammaticality | G |
| 15 | `webnlg2020_fluency` | WebNLG 2020 / data-to-text | Correctness, Data Coverage, **Fluency**, Relevance, **Text Structure** | Fluency | C + Cl + N |
| 16 | `webnlg2020_text_structure` | WebNLG 2020 / data-to-text | Correctness, Data Coverage, **Fluency**, Relevance, **Text Structure** | Text Structure | G + C |
| 17 | `protagolabs_gec_grammaticality` | Protagolabs / GEC (BEA-2019) | **Grammaticality**, Over-correction, Semantics | Grammaticality | G |
| 18 | `tmu_gfm_grammaticality` | TMU-GFM / GEC (CoNLL-2013) | **Fluency**, **Grammaticality**, Meaning Preservation | Grammaticality | G + Cl |
| 19 | `tmu_gfm_fluency` | TMU-GFM / GEC (CoNLL-2013) | **Fluency**, **Grammaticality**, Meaning Preservation | Fluency | N |
| 20 | `parabank_fluency` | ParaBank / paraphrasing | **Fluency**, Semantic Similarity | Fluency | G |
| 21 | `protagolabs_summary_fluency` | Protagolabs / CNN-DM summarization | **Coherence**, Consistency, **Fluency**, Relevance | Fluency | G |
| 22 | `protagolabs_summary_coherence` | Protagolabs / CNN-DM summarization | **Coherence**, Consistency, **Fluency**, Relevance | Coherence | C |
| 23 | `dialsumm_eval_fluency` | DialSummEval / SAMSum dialogue summarization | **Coherence**, Consistency, **Fluency**, Relevance | Fluency | G |
| 24 | `dialsumm_eval_coherence` | DialSummEval / SAMSum dialogue summarization | **Coherence**, Consistency, **Fluency**, Relevance | Coherence | C |
| 25 | `summeval_op_fluency` | SummEval-OP / Amazon opinion summarization | Aspect Coverage, **Coherence**, Faithfulness, **Fluency**, Relevance, Sentiment Consistency, Specificity | Fluency | G + Cl |
| 26 | `summeval_op_coherence` | SummEval-OP / Amazon opinion summarization | Aspect Coverage, **Coherence**, Faithfulness, **Fluency**, Relevance, Sentiment Consistency, Specificity | Coherence | C |
| 27 | `asset_fluency` | ASSET / text simplification | Adequacy, **Fluency**, Simplicity | Fluency | G |
| 28 | `human_likert_fluency` | Human-Likert / text simplification | **Fluency**, Meaning Preservation, Simplicity | Fluency | G |
| 29 | `metaeval_fluency` | Meta-evaluation simplification set | Adequacy, **Fluency**, Simplicity | Fluency | G + N |
| 30 | `protagolabs_simplification_fluency` | Protagolabs / Newsela simplification | **Fluency**, Semantics, Simplicity | Fluency | G + Cl |
| 31 | `samsa_grammaticality` | SAMSA / sentence splitting and simplification | **Grammaticality**, Meaning Preservation, Structural Simplicity | Grammaticality | G |
| 32 | `ELLIPSE__grammar` | ELLIPSE / learner essays | Overall, **Cohesion**, Syntax, Vocabulary, Phraseology, **Grammar**, Conventions | Grammar | G |
| 33 | `ELLIPSE__cohesion` | ELLIPSE / learner essays | Overall, **Cohesion**, Syntax, Vocabulary, Phraseology, **Grammar**, Conventions | Cohesion | C |
| 34 | `HUMAN-CHATGPT-ESSAYS__language_mastery` | Human–ChatGPT Essays (Herbold et al., 2023) / argumentative essays | Topic and completeness, Logic and composition, Expressiveness and comprehensiveness, **Language mastery**, Complexity, Vocabulary and text linking, **Language constructs** | Language mastery | G |
| 35 | `HUMAN-CHATGPT-ESSAYS__language_constructs` | Human–ChatGPT Essays (Herbold et al., 2023) / argumentative essays | Topic and completeness, Logic and composition, Expressiveness and comprehensiveness, **Language mastery**, Complexity, Vocabulary and text linking, **Language constructs** | Language constructs | G |
| 36 | `CoheSentia__coherence_holistic` | CoheSentia / generated stories | **Holistic coherence**, **incremental coherence**; sentence-level coherent/incoherent decision and incoherence-reason codes | Holistic coherence | C |
| 37 | `CoheSentia__coherence_incremental` | CoheSentia / generated stories | **Holistic coherence**, **incremental coherence**; sentence-level coherent/incoherent decision and incoherence-reason codes | Incremental coherence | C |
| 38 | `SummEval__fluency` | Original SummEval / CNN-DM summarization | **Coherence**, Consistency, **Fluency**, Relevance | Fluency | G |
| 39 | `SummEval__coherence` | Original SummEval / CNN-DM summarization | **Coherence**, Consistency, **Fluency**, Relevance | Coherence | C |
| 40 | `JFLEG_test_correction_preference` | JFLEG test / pairwise GEC | preferred human correction; dispreferred learner source | holistic fluency preference | G + N¹ |
| 41 | `MultiBLiMP_eng_minimal_pair_preference` | MultiBLiMP English / minimal pairs | grammatical sentence; minimally different ungrammatical sentence | acceptability preference | G¹ |
| 42 | `StoryCloze_eval_ending_preference` | Story Cloze eval / ending choice | right ending; wrong ending | narrative-ending preference | C¹ |

¹ The pairwise runner stores only the aspect strings `grammar`, `acceptability`, and `coherence`, not a `fluency_categories` tuple. These mappings are explicit interpretations for this documentation. Story Cloze is already marked in code as a **secondary** coherence diagnostic because commonsense plausibility contributes to its label.

## Dataset-level interpretation table

This table is a quick companion for preliminary interpretation of per-eval results. It can help generate hypotheses when an evaluator behaves unusually on one collection—for example, whether unusually strong performance coincides with short templatic outputs, learner writing, or formal argumentative essays. These corpus characteristics are **descriptive context, not causal explanations**: an apparent association in a result plot should be treated as a question for follow-up analysis rather than evidence that register, formality, or length caused the result.

For scalar datasets, “texts used” is the number of **distinct scored candidate texts**, not the number of text–dimension records: the same 270 Herbold et al. essays, for example, are evaluated under two dimensions. For pairwise diagnostics, the table reports evaluation pairs and, where applicable, their source count. Length is the mean number of whitespace-separated words in the exact scored candidate text, recomputed from the suite data unless the detailed profile says otherwise. Register and formality are concise qualitative summaries of the source descriptions and observed texts; they were not independently annotated for every item and should not be treated as measured covariates.

| Dataset | Labels/dimensions used | Fluency dimension | Texts used | Mean length | Register / text type | Formality |
|---|---|---|---:|---:|---|---|
| Chiang LLM Evaluation | Cohesiveness; Grammaticality | C; G | 378 | 139.5 words | Creative WritingPrompts stories; often first-person, dialogue-heavy, or fantastical | Mixed: informal to literary |
| CoEval | Grammaticality; Coherence; Clarity | G; C; Cl | 200 | 55.8 words | Short ROC-style everyday-event narratives | Neutral to informal |
| HANNA | Coherence | C | 957 | 253.4 words | Human- and machine-written creative WritingPrompts stories | Highly variable |
| The Next Chapter | Fluency; Coherence | G; C | 260 | 127.6 words | Mixture of ROCStories, creative fiction, and tokenized news continuations | Highly mixed |
| PPLM | Fluency | G; Cl | 2,499 | 63.9 words | Attribute- or topic-steered open-ended web-style continuations | Informal to neutral |
| E2E NLG | Naturalness | N | 2,723 | 24.2 words | Short, often templatic restaurant descriptions | Neutral informational |
| INLG 2016 | Naturalness | N | 1,133 | 17.3 words | Usually one-sentence, templatic restaurant descriptions | Neutral informational |
| RankME | Naturalness | N | 180 | 14.4 words | Very short E2E restaurant descriptions | Neutral informational |
| WebNLG 2017 | Fluency; Grammaticality | N; G | 1,737 | 19.9 words | DBpedia triple verbalizations across encyclopedic domains | Neutral to formal factual |
| WebNLG 2020 | Fluency; Text Structure | G; C; Cl; N | 2,468 | 22.0 words | RDF-to-text realizations in multiple encyclopedic domains | Neutral to formal factual |
| Protagolabs GEC | Grammaticality | G | 285 | 19.0 words | Corrected learner-English sentences from BEA-2019 outputs | Mostly formal/academic |
| TMU-GFM | Grammaticality; Fluency | G; Cl; N | 4,217 | 20.0 words | Corrected sentences from learner argumentative essays | Formal/academic |
| ParaBank | Fluency | G | 5,550 | 22.2 words | Automatically generated paraphrases; often conversational or subtitle-like | Informal to variable |
| Protagolabs summarization | Fluency; Coherence | G; C | 392 | 58.4 words | Extractive and abstractive CNN/DailyMail news summaries | Neutral to formal journalistic |
| DialSummEval | Fluency; Coherence | G; C | 1,333 | 22.9 words | Summaries of messenger-style dialogues; sometimes telegraphic | Informal content; informal to neutral realization |
| SummEval-OP | Fluency; Coherence | G + Cl; C | 416 | 87.3 words | Multi-review Amazon product-opinion summaries | Semi-formal explanatory |
| ASSET | Fluency | G | 100 | 15.2 words | Single-sentence Wikipedia simplifications | Neutral encyclopedic |
| Human-Likert simplification | Fluency | G | 112 | 16.3 words | Single-sentence TurkCorpus/WikiLarge simplifications | Neutral encyclopedic |
| Meta-evaluation simplification | Fluency | G + N | 588 | 17.0 words | Short Wikipedia/TurkCorpus simplifications | Neutral encyclopedic |
| Protagolabs simplification | Fluency | G + Cl | 357 | 18.9 words | Newsela news/educational sentence simplifications | Neutral educational |
| SAMSA | Grammaticality | G | 493 | 25.4 words | WikiSmall/PWKP simplifications emphasizing sentence splitting | Neutral encyclopedic |
| ELLIPSE | Grammar; Cohesion | G; C | 6,482 | 427.8 words | English-learner state-assessment essays, often argumentative school writing | Intended formal; realized quality varies widely |
| Human–ChatGPT Essays (Herbold et al., 2023) | Language Mastery; Language Constructs | G | 270 | 283.2 words | Human, GPT-3.5, and GPT-4 essays on educational/social propositions | Formal academic; model texts often formulaic |
| CoheSentia | Holistic Coherence; Incremental Coherence | C | 49 | 121.1 words | Automatically generated short creative stories | Informal/creative |
| Original SummEval | Fluency; Coherence | G; C | 1,600 | 63.0 words | Extractive and abstractive CNN/DailyMail news summaries | Neutral to formal journalistic |
| JFLEG test | Correction preference | G + N | 2,582 pairs from 748 learner sources | 19.7 words (preferred correction) | Isolated TOEFL learner-English expository/opinion sentences | Neutral to formal; variable proficiency |
| MultiBLiMP English | Grammatical minimal-pair preference | G | 770 pairs | 21.9 words (grammatical member) | Dependency-treebank sentences retaining treebank tokenization | Mixed: conversational to edited informational |
| Story Cloze eval | Narrative-ending preference | C | 468 pairs | 42.5 words (context + right ending) | Five-sentence crowdwritten everyday ROCStories | Simple and informal |

The detailed profiles below provide medians, ranges, examples, annotation definitions, and source-specific limitations. When using this table alongside result plots, especially useful follow-ups include checking score association with length within the noteworthy eval, stratifying by generation system where that identifier exists, and comparing the finding with another dataset of similar register but different task construction.

## Annotation definitions and corpus profiles

Profiles are grouped by source collection because multiple eval dimensions often use exactly the same texts. Each profile applies to every eval named in its heading.

### 1. Chiang LLM Evaluation — `chiang_cohesiveness`, `chiang_grammaticality`

**Relevant criteria.** Annotators rated how well the story-fragment sentences fit together (**Cohesiveness**, C) and how grammatically correct the fragment is (**Grammaticality**, G), on 1–5 scales. These are the questions preserved verbatim in NLG-Eval. The source also labels likability and prompt relevance, which are deliberately outside this suite.

**Texts.** `n = 378` per dimension; mean 139.5 words, median 142, range 92–150. The material is WritingPrompts story generation: informal to literary narrative prose, often first-person, dialogue-heavy, fantastical, and deliberately creative. The register varies more than edited news or encyclopedic prose.

**IAA.** The publication reports Krippendorff's alpha and exact agreement separately by text origin. Human-written stories: grammaticality α = .33 / exact agreement 20.5%; cohesiveness α = .32 / 27.0%. GPT-2 stories: grammaticality α = .10 / 19.5%; cohesiveness α = .14 / 17.0%. These are low item-level agreements and should not be hidden by averaging the three ratings.

**Examples (shortened from checked-in records).**

> “Me not that kind of Orc!” … “Go! Now! Were running behind schedule!” … I rush to the site. I hear a whoosh of wind.

> The Talisman is a traditional, ornate, golden Aztec-ish figure. It has six slender arms and a long crocodile tail. I've had it for a couple of months…

**Official sources:** [paper](https://arxiv.org/abs/2305.01937), [authors' repository](https://github.com/d223302/LLM-Evaluation), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 2. CoEval — `coeval_grammaticality`, `coeval_coherence`, `coeval_clarity`

**Relevant criteria.** On 1–5 scales, **Grammaticality** asks whether the generated story is grammatically correct; **Coherence** requires logical flow and closure; **Clarity** requires easy understanding without confusing or ambiguous elements. Character development, engagement, length, and relevance are also present but excluded.

**Texts.** `n = 200` per dimension; mean 55.8 words, median 52.5, range 15–167. These are short, prompted ROC-style everyday-event stories, often around four sentences. They are accessible, neutral narrative prose rather than formal exposition.

**IAA.** CoEval reports overall Krippendorff's α = .64 for its human-only condition and .71 for human-in-the-loop evaluation. The paper does not report a separate α for each of these three labels in the 200-record ROC subset; presenting .64 as “clarity IAA,” for example, would therefore overstate the evidence.

**Examples.**

> Molly decided to drive her mom's car to the grocery store to pick up some ingredients for dinner. … Molly returned home to cook dinner with her mom.

> Molly decided to drive her mom's car to work today. She had never driven it before, but she thought it would be a fun challenge.

**Official sources:** [paper](https://arxiv.org/abs/2310.19740), [authors' repository](https://github.com/qtli/CoEval), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 3. HANNA — `hanna_coherence`

**Relevant criterion.** **Coherence** asks how much the generated story makes sense (1–5). Complexity, empathy, engagement, relevance, and surprise are also annotated but not selected.

**Texts.** `n = 957`; mean 253.4 words, median 224, range 17–880. They are WritingPrompts stories from human writers and automatic story-generation systems. The collection contains long-form creative prose, dialogue, unusual premises, and substantial variation in editing, tone, and formality.

**IAA.** The paper uses ICC(2,k), appropriate to the average of three random raters. Coherence has ICC(2,k) = .29, 95% CI [.10, .48]. For context, the six criteria range from .28 to .56. This is weak reliability for the particular averaged score, not a kappa or simple percent agreement.

**Examples.**

> 3,000 years have I been fighting. Every morning, the raccoons scratch at my eyes. Every evening, the skunks spray me…

> When Tyler entered the ward, his daughter Valerie was already fast asleep, her frail body no match for the potent cocktail of drugs coursing through her veins.

**Official sources:** [paper](https://aclanthology.org/2022.coling-1.509/), [authors' repository](https://github.com/dig-team/hanna-benchmark-asg/tree/coling), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 4. The Next Chapter — `nextchapter_fluency`, `nextchapter_coherence`

**Relevant criteria.** **Fluency** asks how grammatically correct a generated story is (therefore G here); **Coherence** asks how well its sentences fit together (C). Interestingness, logicality, and relatedness are available but excluded. Ratings use 1–5.

**Texts.** `n = 260` per dimension: 120 ROCStories, 80 WritingPrompts, and 60 CNN/DailyMail continuations. Mean 127.6 words, median 78, range 10–988. Consequently this single eval mixes short everyday stories, creative fiction, and lower-cased/tokenized news-like continuations; its register and length distribution are unusually heterogeneous.

**IAA.** The paper reports one-vs-rest Pearson correlation / total exact agreement for three MTurk ratings. ROC: fluency .64 / 17.24%, coherence .81 / 24.98%. WritingPrompts: .51 / 18.37%, .70 / 17.01%. CNN/DailyMail: .46 / 15.13%, .54 / 12.61%. A separate in-house annotation has ROC .42 / 38.57% and .54 / 25.00%; WritingPrompts .36 / 10.00% and .57 / 10.00%; CNN/DailyMail .36 / 17.14% and .41 / 10.71%, for fluency and coherence respectively. These two rater pools must not be combined.

**Examples.**

> the girl … was detained on tuesday after she and a group of other migrants walked around a vehicle checkpoint and turned themselves in to border patrol agents…

> the child appeared by dna tests sunday, a day after authorities said her supposed mother was found safe at a las vegas … rest stop.

**Official sources:** [paper](https://arxiv.org/abs/2301.09790), [authors' repository](https://github.com/ZhuohanX/TheNextChapter), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 5. PPLM — `pplm_fluency`

**Relevant criterion.** **Fluency** asks whether the generation is free of grammatical errors, formatting problems, fragments, or missing components that make it hard to read (1–5). Because the instruction combines well-formedness with ease of reading, the registry maps it to G + Cl.

**Texts.** `n = 2,499`; mean 63.9 words, median 66, range 9–99. These are open-ended continuations steered toward attributes or topics by Plug and Play Language Models. They resemble informal web prose and short expository or autobiographical paragraphs; truncation and unfinished endings are common and are part of what the label measures.

**IAA.** No numerical IAA for this fluency annotation is reported in the original publication/release.

**Examples.**

> The relationship between the two cities is already well defined in the way they interact with the rest of the country. But with an estimated $1.1 billion…

> The relationship between my girlfriend and I is pretty good. We have been together for about a year now but we are both in college.

**Official sources:** [paper](https://arxiv.org/abs/1912.02164), [authors' annotation release](https://github.com/uber-research/PPLM/tree/master/human_annotation), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 6. E2E NLG — `e2e_naturalness`

**Relevant criterion.** **Naturalness** asks whether the utterance could have been produced by a native speaker. The original evaluation uses relative ranking / RankME-style magnitude estimation rather than an ordinary five-point rating. Overall quality is the other source label and is excluded.

**Texts.** `n = 2,723`; mean 24.2 words, median 25, range 5–54. These are short restaurant-domain data-to-text utterances realizing meaning representations such as name, area, price, food, and family-friendliness. Register is neutral informational English, with many templatic outputs and some sentence fragments.

**IAA.** The E2E evaluation report does not provide a conventional numerical IAA for naturalness.

**Examples.**

> The city centre is a coffee shop called Blue Spice.

> In city centre is a coffee shop called Blue Spice.

**Official sources:** [full E2E evaluation report](https://arxiv.org/abs/1901.07931), [official evaluation repository](https://github.com/tuetschek/e2e-eval), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 7. INLG 2016 — `inlg16_naturalness`

**Relevant criterion.** **Naturalness** asks whether the utterance is natural—for example, whether a native speaker could have produced it—on the source study's six-point scale. Informativeness and phrasing are also annotated but excluded.

**Texts.** `n = 1,133`; mean 17.3 words, median 16, range 4–73. These are short restaurant-domain system outputs, generally one sentence, in neutral informational register. Their templatic nature is similar to E2E but system and experimental setup differ.

**IAA.** The paper reports Cohen's κ = −.007 (`p = .62`) for naturalness when comparing system developers' self-evaluations with independent crowd evaluations. This is a cross-group agreement analysis, not within-crowd IAA; it should not be described as ordinary rater reliability.

**Examples.**

> Wildwood is a restaurant located centrally that is family friendly.

> Wildwood is a restaurant that provide family service.

**Official sources:** [paper](https://aclanthology.org/W16-6644/), [authors' repository](https://github.com/jeknov/INLG_16_submission), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 8. RankME — `rankme_naturalness`

**Relevant criterion.** As in E2E, **Naturalness** asks whether a native speaker could have produced the utterance. The annotation is relative magnitude estimation, subsequently represented numerically. Informativeness and overall quality are not used.

**Texts.** `n = 180`; mean 14.4 words, median 16, range 7–23. These are very short E2E restaurant-domain data-to-text outputs in neutral, largely templatic informational prose.

**IAA.** The paper reports intraclass correlations for naturalness under two experimental setups: Likert .07 / .12; plain magnitude estimation −.03 / .27; RankME .11 / .42 (the latter two Setup-2 non-Likert estimates are marked significant). The suite uses the RankME collection, but the compiled JSONL does not encode the setup identifier; .42 is therefore the closest published reliability figure, not a guaranteed row-level property of every retained record.

**Examples.**

> Blue Spice is a pub in the city centre.

> Blue Spice is a riverside restaurant located in the riverside area.

**Official sources:** [paper](https://aclanthology.org/N18-2012/), [authors' repository](https://github.com/jeknov/RankME), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 9. WebNLG 2017 — `webnlg2017_fluency`, `webnlg2017_grammaticality`

**Relevant criteria.** **Fluency** asks whether the text sounds fluent and natural (N); **Grammaticality** asks whether it has spelling or grammatical errors (G). Semantic adequacy is present but excluded.

**Texts.** `n = 1,737` per dimension in a direct scan of the current JSONL; mean 19.9 words, median 18, range 3–99. Outputs verbalize DBpedia RDF triples and cover multiple encyclopedic domains. They are short, neutral/factual sentences or small paragraphs; malformed names, omissions, and fragments occur.

**IAA.** No numerical IAA for these human labels is reported in the official challenge paper/site.

**Examples.**

> on february 28, 1966.

> elliot see died on 1966-02-28.

**Audit warning.** The checked-in manifest says 1,738 records, while a fresh loader-equivalent scan finds 1,737. The metadata CSV declares a 1–3 scale, but the checked-in human scores include values such as 5.0. Results should not be interpreted as preserving the publication's raw scale until this provenance mismatch is resolved.

**Official sources:** [challenge paper](https://aclanthology.org/W17-3518/), [official challenge site](https://synalp.gitlabpages.inria.fr/webnlg-challenge/challenge_2017/), [suite records](../data/benchmarks/NLG-Eval.jsonl), [checked-in manifest](../data/benchmarks/english_fluency_suite_manifest.json).

### 10. WebNLG 2020 — `webnlg2020_fluency`, `webnlg2020_text_structure`

**Relevant criteria.** **Fluency** asks whether the text progresses naturally, forms a coherent whole, and is easy to understand (C + Cl + N). **Text Structure** asks whether it is grammatical, well structured, and acceptable English (G + C). Correctness, data coverage, and relevance are excluded.

**Texts.** `n = 2,468` per dimension in a direct scan; mean 22.0 words, median 20, range 3–73. Like 2017, these are short RDF-to-text realizations in factual/encyclopedic register, but from the bilingual, bidirectional WebNLG+ 2020 task.

**IAA.** No numerical IAA for these dimensions is reported in the official task paper/site.

**Examples.**

> The Motor sport of Vision is in Fawkham.

> MotorSport Vision is located in Fawkham.

**Audit warning.** The checked-in manifest says 2,469 rather than the 2,468 valid records found by a fresh scan. The metadata CSV describes a 0–100 scale, while values in the JSONL look approximately 1–5 (for example 4.4). Treat the compiled numeric scale as unresolved.

**Official sources:** [challenge paper](https://aclanthology.org/2020.webnlg-1.22/), [official challenge site](https://webnlg-challenge.loria.fr/challenge_2020/), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 11. Protagolabs GEC — `protagolabs_gec_grammaticality`

**Relevant criterion.** In the original paper, **Grammaticality** is an error count: how many errors remain in the correction, whether inherited or newly introduced. Its scale is **0 (best) to 3 (worst)**. Semantics and over-correction are also annotated but excluded.

**Texts.** `n = 285`; mean 19.0 words, median 17, range 1–47. They are corrected learner-English sentences from BEA-2019 outputs: short, mostly formal or academic prose, with residual grammatical and editorial errors.

**IAA.** Human system-ranking interval Krippendorff's α = 1.00 for grammaticality. This is agreement among three annotators after their 100 item scores were averaged into rankings of four systems; it is **not** item-level score reliability. Including GPT-4 yields α = .83.

**Examples.**

> The other point is when it comes to personal matters. People usually tend to keep it private…

> The other point is when it comes to personal matters, people usually intend to keep it private…

**Critical audit warning.** Some checked-in NLG-Eval values are outside 0–3 (the first record has `[3, 1, 5]`). More importantly, the suite currently averages the values and correlates them without reversing direction, although the publication defines lower as better. This eval should not be used for scientific conclusions until the values' provenance and polarity are repaired.

**Official sources:** [paper](https://aclanthology.org/2023.emnlp-main.543/), [authors' repository](https://github.com/protagolabs/seq2seq_llm_evaluation), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 12. TMU-GFM — `tmu_gfm_grammaticality`, `tmu_gfm_fluency`

**Relevant criteria.** **Grammaticality** assesses both grammatical correctness and comprehensibility: 4 is perfectly grammatical, descending through comprehensible errors to 1 for incomprehensible and 0 for “other.” The separate **Fluency** judgment asks how natural the correction sounds to native speakers, from 4 (extremely natural) to 1 (extremely unnatural). Meaning preservation is excluded.

**Texts.** `n = 4,217` per dimension; mean 20.0 words, median 19, range 3–85. These are corrected CoNLL-2013 learner essays. Most are single formal/academic argumentative sentences, with both native-like corrections and residual GEC errors.

**IAA.** No numerical IAA is reported in the original TMU-GFM paper/release.

**Examples.**

> After all, there will be an endless battle between the technology and human mentality.

> In addition, the national defense will be endangered if this tracking proposal is launched.

**Official sources:** [paper](https://aclanthology.org/2020.coling-main.573/), [authors' repository](https://github.com/tmu-nlp/TMU-GFM-Dataset), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 13. ParaBank — `parabank_fluency`

**Relevant criterion.** The original authors operationalize fluency by asking annotators to **flag** outputs that are completely nonsensical or indisputably ungrammatical; a sentence is counted as problematic if at least one independent annotator flags it. Semantic similarity is a separate 0–100 judgment and is not used.

**Texts.** `n = 5,550`; mean 22.2 words, median 20, range 1–200. These are automatically generated English paraphrases, often conversational or subtitle-like, ranging from short sentences to occasional long strings. The input source and paraphrase are separate; only the paraphrase is scored by this eval.

**IAA.** The paper reports at least three independent judgments per sentence and attentiveness filtering, but no numerical IAA for the fluency flag.

**Examples.**

> If my son came back and your father wasn't home, it'd be better if you didn't let him in.

> If my boy came back and your father wasn't home, it would be better if you didn't allow him inside.

**Critical audit warning.** The original label is a flag, whereas the checked-in records contain repeated values that look like five-point ratings (for example eight `5.0`s). The metadata CSV's “0–2” description also matches neither representation. Until provenance is established, the compiled score must not be called the original ParaBank fluency label.

**Official sources:** [paper](https://arxiv.org/abs/1901.03644), [authors' evaluation release](https://github.com/decompositional-semantics-initiative/ParaBank-Eval-Data), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 14. Protagolabs summarization — `protagolabs_summary_fluency`, `protagolabs_summary_coherence`

**Relevant criteria.** Following SummEval on 1–5 scales, **Fluency** assesses sentence quality, grammaticality, and readability; **Coherence** assesses whether sentences collectively form a well-structured, logically ordered summary. Consistency and relevance are excluded.

**Texts.** `n = 392` per dimension; mean 58.4 words, median 54, range 13–121. These are extractive and abstractive CNN/DailyMail news summaries from a gold reference and three model families. Register is compressed journalistic prose, usually a short paragraph or bullet-like sequence.

**IAA.** Human system-ranking interval Krippendorff's α = .88 for fluency and 1.00 for coherence. With GPT-4 included, α = .63 and .72 respectively. As above, these are four-system rankings derived from mean scores, not item-level IAA.

**Examples.**

> Mats Hummels has two years left on deal but is considering his future. Manchester United reportedly ready to pay £30million…

> Borussia Dortmund have offered Mats Hummels a contract extension. Hummels is considering his future at the Bundesliga club.

**Official sources:** [paper](https://aclanthology.org/2023.emnlp-main.543/), [authors' repository](https://github.com/protagolabs/seq2seq_llm_evaluation), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 15. DialSummEval — `dialsumm_eval_fluency`, `dialsumm_eval_coherence`

**Relevant criteria.** On 1–5 scales, **Fluency** assesses sentence-level quality and absence of grammatical/formatting problems; **Coherence** assesses whether sentences fit together, are ordered logically, and form a well-structured summary. Consistency and relevance are excluded.

**Texts.** `n = 1,333` per dimension; mean 22.9 words, median 17, range 2–164. The target texts summarize SAMSum messenger-style dialogues. They are short and often informal in content, but summaries themselves range from compressed telegraphic lower-case output to conventional sentences.

**IAA.** Krippendorff's α: fluency .6782, coherence .7564. The paper also reports consistency .6709 and relevance .5621, but those labels are not used here.

**Examples.**

> dorothea is having a birthday dinner in the town with tom. elena is seeing dorothea at her party on saturday.

> Dorothea: Yes, I'm gonna meet Tom and we're going to eat something in the town :) Elena: Cool! <3 …

The second example illustrates that some retained targets appear dialogue-like rather than summary-like; this is visible in the official compiled records and should be considered when interpreting fluency.

**Official sources:** [paper](https://aclanthology.org/2022.naacl-main.418/), [authors' repository](https://github.com/kite99520/DialSummEval), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 16. SummEval-OP — `summeval_op_fluency`, `summeval_op_coherence`

**Relevant criteria.** **Fluency** concerns grammatical correctness and ease of reading (G + Cl); **Coherence** concerns organization, logical flow, and relations among ideas (C). The collection additionally labels relevance, faithfulness, aspect coverage, sentiment consistency, and specificity.

**Texts.** `n = 416` per dimension; mean 87.3 words, median 84, range 38–204. These are summaries of sets of Amazon product reviews. They are consumer-domain opinion syntheses in semi-formal explanatory prose, often covering fit, quality, price, and recommendations.

**IAA.** Krippendorff's α improved between annotation rounds: fluency .55 → .84 and coherence .43 → .73. Other Round-I → Round-II values are relevance .50 → .79, faithfulness .63 → .86, aspect coverage .64 → .82, sentiment consistency .41 → .78, and specificity .34 → .76. Suite records should be tied to a round before choosing which figure applies; the compilation does not expose round in its normalized row.

**Examples.**

> Nice boots but run a bit narrow. They look great but I think the quality has come down over the years.

> These boots are well made and comfortable, with high-quality leather that lasts for many years. They may require some breaking in…

**Official sources:** [paper](https://arxiv.org/abs/2402.11683), [authors' repository](https://github.com/tjsiledar/SummEval-OP), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 17. ASSET — `asset_fluency`

**Relevant criterion.** **Fluency** asks whether the simplification is fluent English without grammatical errors, using direct-assessment scores on a 0–100 scale. Adequacy and simplicity are excluded.

**Texts.** `n = 100`; mean 15.2 words, median 14, range 3–35. These are single-sentence Wikipedia simplifications for TurkCorpus/WikiLarge source sentences. Register is neutral encyclopedic prose, though automatic outputs may be awkward or fragmentary.

**IAA.** The authors simulate split-half rater groups and report quadratic-weighted Cohen's κ = .687 ± .028 for fluency (.686 ± .030 meaning preservation; .628 ± .032 simplicity).

**Examples.**

> Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award…

> After the drummers are dancers, who often play the Sogo (a small drum that makes almost no sound).

**Official sources:** [paper](https://aclanthology.org/2020.acl-main.424/), [official repository](https://github.com/facebookresearch/asset), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 18. Human-Likert simplification — `human_likert_fluency`

**Relevant criterion.** **Fluency** asks how fluent the simplified text is. Meaning preservation and simplicity are separate labels. The original experiment describes five-point Likert ratings with about 30 ratings per aspect and 100 unique simplifications; NLG-Eval's metadata instead describes 0–100 and 12–35 annotators.

**Texts.** `n = 112` in the checked-in compilation; mean 16.3 words, median 16, range 4–38. These are TurkCorpus/WikiLarge single-sentence simplifications in encyclopedic register.

**IAA.** The paper discusses lower agreement for Human-Likert than System-Likert but does not provide a label-specific numerical IAA for these fluency items.

**Examples.**

> Many people who receive the Kate Greenaway medal also win the Colin Mears award. its worth £5000.

> Drummers playing a Sogo (a tiny drum that makes little sound) are followed by elaborately choreographed dancers.

**Audit warning.** The compiled count (112) and declared scale do not transparently match the publication's 100-item/five-point description. Treat the local representation as a transformed derivative until its conversion is documented.

**Official sources:** [paper](https://arxiv.org/abs/2104.07560), [authors' released archive](http://dl.fbaipublicfiles.com/questeval/simplification_human_evaluations.tar.gz), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 19. Meta-evaluation simplification set — `metaeval_fluency`

**Relevant criterion.** **Fluency** directs the rater to consider grammar, spelling, and whether the text reads naturally, while ignoring capitalization (G + N). Adequacy and simplicity are excluded. Scores are direct assessments represented on 0–100.

**Texts.** `n = 588`; mean 17.0 words, median 16, range 1–49. These are short Wikipedia/TurkCorpus simplifications in neutral encyclopedic register.

**IAA.** The paper reports overall reliability for its Simplicity-DA judgments (ICC = .386; split-group Spearman = .607 ± .026), not label-specific IAA for this fluency subset. Those figures must not be relabeled “fluency IAA.”

**Examples.**

> Prunk is a member of Institute of European History in Mainz. He was also a member of the Center for European Integration Studies in Bonn.

> In return, Rollo swore fealty to Charles, converted to Christianity, and set out to defend the north of France…

**Official sources:** [paper](https://aclanthology.org/2021.cl-4.28/), [authors' HIT designs](https://github.com/feralvam/metaeval-simplification/tree/main/HIT_designs), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 20. Protagolabs simplification — `protagolabs_simplification_fluency`

**Relevant criterion.** **Fluency** asks whether the simplification is understandable and grammatical on a 1–5 scale (G + Cl). Semantics and simplicity are excluded.

**Texts.** `n = 357`; mean 18.9 words, median 17, range 2–74. These are Newsela sentence simplifications: short news/educational prose written for reduced reading difficulty, with outputs from a gold reference and several language models.

**IAA.** Human system-ranking interval Krippendorff's α = 1.00 for fluency; including GPT-4 yields α = .72. This is ranking agreement over four systems after item-score aggregation, not item-level IAA.

**Examples.**

> He makes albums for the music company to sell.

> He records music for them, and they put out and sell his albums.

**Official sources:** [paper](https://aclanthology.org/2023.emnlp-main.543/), [authors' repository](https://github.com/protagolabs/seq2seq_llm_evaluation), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 21. SAMSA — `samsa_grammaticality`

**Relevant criterion.** **Grammaticality** asks whether the simplified text is grammatical, on a three-point scale. Meaning preservation and structural simplicity are excluded.

**Texts.** `n = 493`; mean 25.4 words, median 25, range 5–62. These are WikiSmall/PWKP simplification outputs designed especially to test sentence splitting. Register is neutral encyclopedic prose, sometimes with mechanical splitting artifacts.

**IAA.** For grammaticality (question Qa), the paper reports total absolute agreement .58, sentence-level quadratic-weighted κ = .56, and sentence-level Spearman ρ = .63 (system-level ρ = .92). The measures use different aggregation levels and should not be collapsed into one number.

**Examples.**

> Engineering has expanded the genes available to breeders to use in making germlines for new crops.

> The name jadgal applies to groups. Groups still speak the jadgali language in the region of iran and pakistan.

**Official sources:** [paper](https://aclanthology.org/N18-1063/), [authors' repository](https://github.com/eliorsulem/SAMSA), [suite records](../data/benchmarks/NLG-Eval.jsonl).

### 22. ELLIPSE — `ELLIPSE__grammar`, `ELLIPSE__cohesion`

**Relevant criteria.** The five-level analytic rubric defines **Grammar** as command of grammar and usage, from errors throughout (1) to few/no errors (5). **Cohesion** covers organization and linguistic links across sentences/paragraphs—reference, transitions, conjunctions, repetition/idea overlap, and anaphora—from absent/unsuccessful control (1) to consistently controlled varied use (5). The authors explicitly distinguish grammar/morphology from syntax. Overall proficiency, syntax, vocabulary, phraseology, and conventions are not used by the present suite.

**Texts.** `n = 6,482` distinct essays (`12,964` eval records across two dimensions); mean 427.8 words, median 398, range 14–1,274. They are US state-assessment essays by English learners in grades 8–12, responding to 29 independent prompts. Much is argumentative school writing addressed to a teacher or principal. The intended register is formal school prose, but proficiency, spelling, punctuation, organization, and length vary widely.

**IAA.** Each essay was scored by two trained raters. Pre-adjudication Cohen's κ: overall .599, cohesion .562, syntax .559, vocabulary .518, phraseology .561, grammar .593, conventions .580. A Many-Facet Rasch analysis found essay reliability .94 and rater-severity reliability .99 after unreliable texts were pruned to the final 6,482; scale reliability was reported as 100%. The paper itself characterizes the initial kappas below .60 as lower than expected, so they should not be called “excellent.”

**Examples (shortened).**

> Dear, TEACHER_NAME I think phone policy at school should not let students use their phone during class or free time. For several reasons.

> Dear, Principal In my opinion, I think that you should allow students to bring their phones to school, but with the condition, to use them outside their classes.

**Official sources:** [authors' repository and dataset](https://github.com/scrosseye/ELLIPSE-Corpus), [official rubric](https://github.com/scrosseye/ELLIPSE-Corpus/blob/main/ELL_Rubrics.docx), [authors' preprint](https://zenodo.org/records/11217937), [checked-in suite copy](../data/benchmarks/ELLIPSE.csv).

### 23. Human–ChatGPT argumentative essays (Herbold et al., 2023) — `HUMAN-CHATGPT-ESSAYS__language_mastery`, `HUMAN-CHATGPT-ESSAYS__language_constructs`

**Identity and name-collision audit.** This is the evaluation dataset accompanying Herbold et al.'s *A large-scale comparison of human-written versus ChatGPT-generated essays*. It should not be called “ArgEssay”: that name is commonly used for the separate dataset introduced by Bao et al. (2022). Herbold et al. selected 90 topics and student essays from the Stab and Gurevych argumentative-writing corpus, generated matched GPT-3.5 and GPT-4 essays, and collected teacher ratings for the resulting 270 texts.

Bao et al.'s ArgEssay is **not included in this eval suite**. Its [official repository](https://github.com/HITSZ-HLT/AEG) releases prompt–essay pairs but no per-essay human evaluation labels. Its [original publication](https://aclanthology.org/2022.emnlp-main.343/) reports a separate human evaluation of generated outputs on Relevance, Coherence, and Content Richness (1–5; three annotators; 50 sampled prompts), with average Fleiss' κ = .42 across the evaluation, but the official release does not provide the aligned item-level ratings and evaluated system outputs required by this suite. Adding it as a labeled eval would therefore require constructing labels not supplied by the official sources.

**Relevant criteria.** Teachers scored seven criteria from 0 (worst) to 6 (best), with written anchor descriptions. **Language mastery** is the criterion most directly associated in the paper with writing mistakes and correct language use. **Language constructs** captures use of the language's constructions; it is broader and may partly overlap syntax/complexity, so its G mapping should be regarded as a pragmatic project choice, not a pure grammar measure. The other five criteria are listed in the inventory above and excluded.

**Texts.** The file contains 90 matched topics, each with a student essay, a GPT-3.5 essay, and a GPT-4 essay (`n = 270` distinct essays; 540 eval records across two dimensions). Whitespace means by source are: student 339.1 words (range 235–1,029), GPT-3.5 248.0 (191–288), GPT-4 262.5 (215–302). These are English argumentative essays about broad educational/social propositions. They aim for formal academic register. The student writers and teacher raters were not native English speakers; the paper flags this as a limitation. Model essays are more rigidly structured and formulaic.

**IAA.** Cronbach's α by criterion: topic/completeness .95, logic/composition .96, expressiveness/comprehensiveness .95, **language mastery .89**, complexity .92, vocabulary/text linking .97, **language constructs .97**. There were 658 ratings from 111 teachers over 270 essays; most essays received two or three ratings, five only one. Cronbach's α is used by the authors as inter-rater reliability, though the unbalanced design and single-rated items should be remembered.

**Examples (shortened).**

> It is always said that competition can effectively promote the development of economy. In order to survive in the competition, companies continue to improve…

> Education is not only about acquiring knowledge, but also about developing the skills and attitudes necessary to succeed in life.

**Official sources:** [original open-access publication](https://www.nature.com/articles/s41598-023-45644-9), [official replication package and data](https://doi.org/10.5281/zenodo.8343644), [checked-in dataset](../data/benchmarks/human-chatgpt-argumentative-essays.csv).

### 24. CoheSentia — `CoheSentia__coherence_holistic`, `CoheSentia__coherence_incremental`

**Relevant criteria.** **Holistic coherence** is a 1–5 judgment made after reading the complete story. **Incremental coherence** is produced while reading sentence by sentence: the annotator decides whether each new sentence remains coherent in context and identifies reasons for incoherence; a final consensus score is retained by this suite. Reason annotations distinguish failures involving cohesion, consistency, relevance, and other discourse relations. Both final scores map to C, but the protocols test different reading processes.

**Texts.** The suite deliberately uses only the 49-story test file (`n = 49` per protocol); mean 121.1 words, median 123, range 55–207. These are automatically generated short stories in informal/creative narrative register. Many contain local grammatical errors, contradictions, abrupt topic shifts, or implausible event chains by construction or generation failure.

**IAA.** Holistic protocol: ICC = .804, Fleiss' κ = .694, Krippendorff's α = .66. Incremental protocol: ICC = .968, κ = .827, α = .86. For incremental sentence-group annotations, the paper reports ICC / κ / α of .96/.87/.90 for coherence, .96/.87/.88 for cohesion, .91/.81/.86 for consistency, and .95/.69/.76 for relevance.

**Examples.**

> I was curious about the world and I wanted to explore it. So, I set out on an adventure. Unfortunately, I quickly got lost…

> It was the early morning hours of March 19, 2020. The sounds of American and British war planes could be heard overhead…

**Official sources:** [paper](https://aclanthology.org/2023.emnlp-main.324/), [checked-in test release](../data/benchmarks/CohesentiaTestData.json).

### 25. Original SummEval — `SummEval__fluency`, `SummEval__coherence`

**Relevant criteria.** On 1–5 scales, **Fluency** assesses individual-sentence quality, grammar, readability, and absence of formatting problems. **Coherence** assesses the summary as a whole: logical ordering, connection among sentences, and a well-structured body of information. Consistency and relevance are excluded.

**Texts.** The maintained release used by the loader has 100 CNN/DailyMail articles × 16 model summaries = `n = 1,600` per dimension. Scored summaries average 63.0 whitespace words, median 61, range 5–133. They mix extractive and abstractive systems and use compressed news register; some early-system outputs are lower-cased/tokenized, repetitive, or malformed. The source release also contains three expert and five crowd annotations per summary, but the MTEB-normalized rows expose an aggregate scalar label.

**IAA.** Interval Krippendorff's α reported by the paper: crowd .4920; expert Round 1 .4132; expert Round 2 .7127 after discussion/refinement. These are overall across dimensions; no separate fluency/coherence α is provided. The paper notes especially substantial disagreement on relevance and coherence.

**Examples.**

> donald sterling, nba team last year. sterling's wife sued for $2.6 million in gifts. sterling says he is the former female companion…

> Donald Sterling accused Stiviano of targeting extremely wealthy older men. She claimed Donald Sterling used the couple's money to buy Stiviano a Ferrari…

**Official sources:** [paper](https://aclanthology.org/2021.tacl-1.24/), [authors' dataset repository](https://github.com/Yale-LILY/SummEval), [maintained dataset loaded by the suite](https://huggingface.co/datasets/mteb/summeval), [loader](../clumsification_code/evals/standalone_benchmarks.py).

### 26. JFLEG test preference — `JFLEG_test_correction_preference`

**Label and relevance.** The loader pairs each human fluency correction as **preferred** against its learner source as **dispreferred**. Annotators were instructed to make the source sound natural and fluent to an American-English native speaker, fixing grammar, awkward phrasing, spelling, and standard usage while keeping edits conservative and meaning unchanged. This spans G and N; it is not merely a minimal grammar correction.

**Texts.** The test split has 748 learner sentences. Removing unchanged corrections produces `n = 2,582` preference pairs in the current loader. Preferred texts average 19.7 words, median 18, range 4–81. These are isolated TOEFL learner-English sentences over varied topics, usually neutral expository or opinion prose, with four human corrections per source.

**IAA.** The authors explicitly state that there is no clear way to quantify agreement among free-form corrections. They report instead that 36% of sentences were corrected identically by at least two participants. This is not chance-corrected IAA. The underlying GUG sources had five crowd grammaticality ratings plus one expert, but those scores are not the pairwise label used here.

**Examples (preferred ⇢ learner source).**

> New technology has been introduced to society. ⇢ New and new technology has been introduced to the society.

> New technology has been introduced into the society. ⇢ New and new technology has been introduced to the society.

**Official sources:** [paper](https://aclanthology.org/E17-2037/), [official JHU dataset release](https://huggingface.co/datasets/jhu-clsp/jfleg), [suite loader](../clumsification_code/evals/benchmark_data.py).

### 27. MultiBLiMP English preference — `MultiBLiMP_eng_minimal_pair_preference`

**Label and relevance.** Each pair contains one automatically generated **grammatical** sentence and a minimally different **ungrammatical** sentence. The final TACL 2026 release covers two subject–verb-agreement types across 101 languages. The current English configuration contains person (`SV-P`, 453 pairs) and number (`SV-#`, 317) agreement, so this eval is narrowly G rather than a comprehensive measure of English acceptability.

**Texts.** `n = 770` pairs; grammatical members average 21.9 whitespace tokens, median 20, range 4–58. Sentences come from dependency-treebank material and retain treebank tokenization (for example `is n't`). Registers therefore reflect the English source treebanks and range from conversational fragments to edited informational prose.

**IAA.** No human IAA applies: pairs are created automatically from Universal Dependencies and UniMorph with rule-based filtering. The release includes an `agreement_certainty` field (754 `+`, 16 `~` in this English split), which is pipeline certainty, not annotator agreement.

**Examples (grammatical ⇢ ungrammatical).**

> Yes, lovely creature, is n't she? ⇢ Yes, lovely creature, am n't she?

> There is no easy answer to the painful issue of abortion. ⇢ There am no easy answer to the painful issue of abortion.

**Official sources:** [paper](https://aclanthology.org/2026.tacl-1.10/), [authors' repository](https://github.com/jumelet/multiblimp), [official dataset release](https://huggingface.co/datasets/jumelet/multiblimp), [suite loader](../clumsification_code/evals/benchmark_data.py).

### 28. Story Cloze ending preference — `StoryCloze_eval_ending_preference`

**Label and relevance.** Each item has a four-sentence context followed by a human-validated **right ending** (preferred) or **wrong ending** (dispreferred). It tests narrative understanding and commonsense inference. It is useful for C only as a secondary diagnostic: an ending can be grammatically fluent and locally connected yet lose because it violates real-world plausibility or the intended script.

**Texts.** The loader's `eval` split has `n = 468` pairs. Context-plus-right-ending texts average 42.5 words, median 42, range 24–61. They are five-sentence ROCStories: short, simple, informal everyday narratives written by crowdworkers.

**IAA.** Candidate endings were rated in `{−1, 0, 1}` and the released cloze items retain only cases whose right ending received all 1s and wrong ending all 0s. The paper calls the sets doubly human-verified and reports 100% human test accuracy. It does not report a chance-corrected IAA coefficient for the retained binary labels.

**Examples (right ⇢ wrong ending; common context abbreviated).**

> James remembered a Visa gift card … James was then able to buy himself food. ⇢ James did not have any money.

> All of Sam's guests left the wild party in disgust. Sam had a terrible hangover the next day. ⇢ Sam became a party planner.

**Official sources:** [original paper](https://aclanthology.org/N16-1098/), [suite dataset release](https://huggingface.co/datasets/lecslab/story_cloze), [suite loader](../clumsification_code/evals/benchmark_data.py).

## Consolidated IAA table

This table is intentionally redundant with the profiles so readers can compare evidence without mistaking unlike coefficients for a common scale.

| Collection | Relevant-label agreement reported by original source | Interpretation / limitation |
|---|---|---|
| Chiang | Human: G α .33, C .32; GPT-2: G .10, C .14; exact agreement also reported | Item-level, 3 raters; low |
| CoEval | overall α .64 human-only, .71 human-in-loop | Not dimension-specific |
| HANNA | coherence ICC(2,k) .29, CI [.10,.48] | Reliability of average of 3 random raters |
| Next Chapter | MTurk Pearson/exact by domain; see profile | One-vs-rest correlation and exact agreement, not α |
| PPLM | not reported | — |
| E2E | not reported | — |
| INLG16 | naturalness κ −.007, p=.62 | Developer self-rating vs crowd, not within-pool IAA |
| RankME | naturalness ICC up to .42 for RankME Setup 2 | Setup identity absent from compiled row |
| WebNLG 2017 | not reported | — |
| WebNLG 2020 | not reported | — |
| Protagolabs GEC | human rank α 1.00; +GPT-4 .83 | System-rank, not item-score agreement |
| TMU-GFM | not reported | — |
| ParaBank | not reported; ≥3 judgments and filtering | Free-standing numeric coefficient absent |
| Protagolabs summary | fluency rank α .88, coherence 1.00 | System-rank, not item-score agreement |
| DialSummEval | fluency α .6782, coherence .7564 | Item annotation reliability |
| SummEval-OP | Round I/II: fluency .55/.84; coherence .43/.73 | Compiled rows lack round provenance |
| ASSET | fluency quadratic-weighted κ .687 ± .028 | Simulated split-half rater groups |
| Human-Likert | no label-specific numerical IAA | Publication discusses relative agreement only |
| Metaeval simplification | no fluency-specific IAA | Overall Simplicity-DA ICC .386 is not fluency IAA |
| Protagolabs simplification | human fluency rank α 1.00; +GPT-4 .72 | System-rank, not item-score agreement |
| SAMSA | grammaticality: absolute .58; QWK .56; sentence ρ .63 | Different agreement/aggregation concepts |
| ELLIPSE | cohesion κ .562; grammar .593; final essay MFRM reliability .94 | Two raters; pre-adjudication kappas; unreliable texts pruned |
| Human–ChatGPT Essays (Herbold et al., 2023) | language mastery α .89; language constructs .97 | Cronbach α under unbalanced 1–3-rater design |
| CoheSentia | holistic ICC/.κ/α .804/.694/.66; incremental .968/.827/.86 | Protocol-level agreement |
| SummEval | crowd α .4920; experts .4132 → .7127 | Overall, not dimension-specific |
| JFLEG | 36% identical correction by ≥2 annotators | Authors say conventional agreement is unclear |
| MultiBLiMP | not applicable | Automatically generated labels |
| Story Cloze | unanimous filtering; 100% human accuracy | No chance-corrected IAA reported |

## Known validity and reproducibility issues

These are part of the dataset documentation, not recommendations to quietly “clean” the data.

1. **NLG-Eval is a transformed compilation.** Its authors merged semantically similar aspect names and manually supplied missing evaluation criteria. A criterion string in the JSONL may therefore be a NLG-Eval reconstruction rather than verbatim original annotation text. Original publications take precedence where they disagree.
2. **Scale mismatches are real.** WebNLG 2017, WebNLG 2020, Protagolabs GEC, ParaBank, and Human-Likert show conflicts between publication, metadata CSV, and observed JSONL values. The Protagolabs GEC polarity is also opposite to the suite's generic “higher-is-better” correlation assumption.
3. **Manifest counts are stale by four NLG dimension-records.** The checked-in manifest records 1,738 for each WebNLG-2017 dimension and 2,469 for each WebNLG-2020 dimension; direct loader-equivalent scans find 1,737 and 2,468. This changes the 31-dimension NLG total from 37,983 in the manifest to 37,979 valid records.
4. **IAA is often not the reliability of the actual scalar used here.** Several sources report system-rank agreement or an overall coefficient, while the suite evaluates correlations over per-item aggregate scores. Those figures are context, not direct uncertainty estimates for the local vector.
5. **Repeated texts create dependence.** The same target is commonly scored under multiple dimensions, and some collections include several outputs for the same prompt/source. Confidence intervals that assume independent rows will be too optimistic unless grouped by source/prompt.
6. **Residual domain and register confounding remains, but the reporting design substantially mitigates it.** “Fluency” covers creative stories, controlled continuations, restaurant descriptions, learner corrections, paraphrases, summaries, and simplifications. The study reports every eval separately and also reports separate grammaticality, coherence, clarity, and naturalness aggregates; it does not rely on the overall aggregate as its substantive construct-level result. The overall score in Table 1 is a space-efficient model-selection summary used to choose the three FE configurations shown in the main-paper aspect table. These choices prevent a strong pooled result from concealing weak performance on individual evals or aspects. Two narrower risks remain: an evaluator may exploit length, tokenization, or generation-system identity **within** an eval, and an aspect-level aggregate may partly reflect the particular mixture of domains assigned to that aspect. Accordingly, per-eval results remain the primary diagnostic evidence, while aspect aggregates should be interpreted as cross-dataset summaries rather than pure, domain-free measurements of a construct.
7. **Aggregate labels discard disagreement.** Standalone loaders expose one scalar mean, and the NLG loader averages `human_score` arrays. Analyses should retain raw rater vectors where valid, report uncertainty, and avoid treating mean labels as error-free ground truth.
8. **The pairwise diagnostics are not scalar-label datasets.** Their output is tie-aware/strict accuracy, not Spearman or Kendall correlation. Story Cloze additionally measures commonsense plausibility; JFLEG preference bundles grammar and native-like rewriting.

## Recommended citation and reporting practice

Any scientific report using this suite should:

- cite both [Themis/NLG-Eval](https://aclanthology.org/2024.emnlp-main.891/) and every constituent dataset actually used;
- state the exact git revision, direct-scan record counts, filtering, and whether preference diagnostics were enabled;
- report results separately for G, C, Cl, and N, and also per source collection rather than only one pooled score;
- disclose the same G/C/Cl/N aggregates for every evaluated FE configuration in supplementary material, including configurations omitted from the main table by the top-three selection rule;
- flag the five unresolved scale/provenance problems above and exclude affected dimensions from headline aggregation until repaired;
- name the IAA statistic rather than saying only “agreement was X”; and
- use grouped resampling by prompt/source document wherever those identifiers are available.

## Local provenance index

- Inclusion/mapping registry: [clumsification_code/evals/benchmark_registry.py](../clumsification_code/evals/benchmark_registry.py)
- Suite runner and pairwise treatment: [clumsification_code/evals/benchmark_runner.py](../clumsification_code/evals/benchmark_runner.py)
- Pairwise dataset construction: [clumsification_code/evals/benchmark_data.py](../clumsification_code/evals/benchmark_data.py)
- Standalone normalization: [clumsification_code/evals/standalone_benchmarks.py](../clumsification_code/evals/standalone_benchmarks.py)
- NLG-Eval records: [data/benchmarks/NLG-Eval.jsonl](../data/benchmarks/NLG-Eval.jsonl)
- NLG-Eval source metadata: [data/benchmarks/NLG-Eval_meta_info.csv](../data/benchmarks/NLG-Eval_meta_info.csv)
- Existing generated manifest: [data/benchmarks/english_fluency_suite_manifest.json](../data/benchmarks/english_fluency_suite_manifest.json)
- Standalone dataset files: [ELLIPSE](../data/benchmarks/ELLIPSE.csv), [Human–ChatGPT argumentative essays](../data/benchmarks/human-chatgpt-argumentative-essays.csv), [CoheSentia test](../data/benchmarks/CohesentiaTestData.json)

## Appendix A. Aspect-level aggregates for all FE configurations

The main-paper aspect table may show only the three FE configurations selected using the overall summary in Table 1 because of the page limit. For transparency, the corresponding aspect-level Spearman correlations for **all five evaluated FE configurations** are reported here. Thus, selection affects presentation in the main paper, not availability of the disaggregated results.

| FE configuration | Grammaticality | Coherence | Clarity | Naturalness | Overall selection summary |
|---|---:|---:|---:|---:|---:|
| E5, traditional perturbations, 4 layers | .210 | .236 | .194 | .169 | .202 |
| E5, traditional perturbations, 5 layers | .189 | .224 | .184 | .149 | .186 |
| E5, mixed perturbations, 5 layers | .230 | .185 | .147 | .225 | .197 |
| E5, clumsy perturbations, 4 layers | .143 | .110 | .074 | .133 | .115 |
| E5, clumsy perturbations, 5 layers | .156 | .150 | .092 | .142 | .135 |

**Provenance and selection note.** Values are the category aggregates stored in the completed [E5 evaluation records](../data/evals/E5.jsonl), rounded to three decimals. Where the results file contains repeated completed records for the same configuration, this table uses the latest record rather than treating reruns as additional models. The “Overall” column is retained only to make the main-paper selection procedure reproducible; it is the mean of the four category aggregates, not a correlation obtained by pooling all text–label pairs across datasets.
