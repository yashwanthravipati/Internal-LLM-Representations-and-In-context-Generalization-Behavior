# Neuroscience connections

This is a course on **biological and artificial intelligence**, and the write-up is judged partly on how well it links the two. Each analysis in this project is borrowed from, or speaks back to, a specific line of neuroscience work. Use this table when writing the report's introduction and discussion.

| Neuroscience idea | Key work | Our analysis | Tickets |
|---|---|---|---|
| **Concept manifolds and few-shot learning in IT cortex.** A downstream neuron with a prototype rule can learn new concepts from a few examples; the geometry of IT population responses predicts which concepts are easy. | Sorscher, Ganguli & Sompolinsky 2022 (*PNAS*) | Exp 1: apply the same theory to LLM residual streams; compare with in-context behavior | T-022–T-024 |
| **Category learning in PFC.** The same stimuli can be grouped by learned category boundaries, and prefrontal neurons come to reflect the learned categories. | Freedman, Riesenhuber, Poggio & Miller 2001 (*Science*) | Exp 2: the same items regrouped by the demonstrated rule | T-030–T-033 |
| **Context-dependent selection.** In PFC, task-irrelevant information is *still represented* but not integrated; selection happens in the dynamics/readout. | Mante, Sussillo, Shenoy & Newsome 2013 (*Nature*) | **H2b (readout selection)**: is the uncued attribute still decodable at the query item? | T-032 |
| **Compression of irrelevant dimensions.** In context-dependent tasks, humans and "rich-regime" networks form orthogonal task representations and compress the irrelevant dimension. | Flesch, Juechems, Dumbalska, Saxe & Summerfield 2022 (*Neuron*) | **H2a (representational shift)**: is the uncued attribute compressed? | T-032 |
| **Geometry of abstraction.** Variables are in an "abstract format" when a decoder trained in one condition generalizes to others (CCGP). | Bernardi, Benna, Rigotti, Munuera, Fusi & Salzman 2020 (*Cell*) | CCGP of the cued attribute across values of the uncued attribute | T-031 |
| **Mixed selectivity.** Nonlinear mixing of task variables enables flexible readout. | Rigotti et al. 2013 (*Nature*) | Interpreting 2×2 coding: are A and B encoded linearly, or mixed? | T-032 |
| **Representational similarity analysis.** Compare representations through dissimilarity matrices, a shared language for brains and models. | Kriegeskorte, Mur & Bandettini 2008 | RSA partial correlations with attribute model RDMs | T-031 |
| **Manifold geometry and classification capacity.** | Chung, Lee & Sompolinsky 2018 (*PRX*) | Background for the manifold view; possible extra metric | — |
| **Global (neuronal) workspace.** A limited-capacity workspace broadcasts selected content to many processes; content in the workspace is reportable. | Baars 1988; Dehaene, Kerszberg & Changeux 1998 (*PNAS*); Gurnee et al. 2026 (J-lens) | Exp 3: does the demonstrated rule enter the "verbalizable" J-space? Does workspace content predict generalization? | T-041–T-043 |
| **Decoding ≠ use.** Decodable information in recordings needn't be used by the circuit; causal perturbation is needed. | Course Project 10 framing; Hewitt & Liang 2019 for probes; optogenetic-style perturbations | Steering/ablation with lens vectors (causal test) | T-044 |
| **Task switching and congruency effects.** Performance differs on stimuli where the two task rules agree vs conflict. | Monsell 2003 (*TiCS*), task-switching literature | Congruent vs incongruent queries; incongruent is the diagnostic condition | T-016, T-032 |
| **Default dimensions in category learning** (e.g. children's shape bias). | Landau, Smith & Jones 1988 | Exp 4: with ambiguous demos, which grouping does the model default to? | T-050 |

## Suggested narrative for the report

1. **Intro.** Humans and monkeys learn new categories from a few examples, and the geometry of IT representations predicts how well (Sorscher). Brains also flexibly regroup the same stimuli by context (Freedman; Mante; Flesch). LLMs show both behaviors in-context. Does the same geometric account explain them?
2. **Exp 1.** A direct test of a neuroscience theory on an artificial system it wasn't designed for.
3. **Exp 2.** The Mante-vs-Flesch question (selection in the readout vs in the representation), asked in a system where we can record every "neuron" at every layer.
4. **Exp 3.** A workspace-style readout: is the inferred rule "reportable" inside the model, and does reportable content carry the generalization signal?
5. **Discussion.** What transfers from neuroscience theory to LLMs and what doesn't, and what LLM results suggest for experiments in brains (e.g. predictions for PFC/IT recordings during rule-based categorization).
