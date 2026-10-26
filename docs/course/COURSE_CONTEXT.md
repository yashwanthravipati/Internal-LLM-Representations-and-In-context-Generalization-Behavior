# Course context

## The class

- **Course:** Neuro 240 / Phys 240, *Biological and Artificial Intelligence*, Fall 2026.
- **Assignment:** the final project.
- **Student:** Yashwanth Ravipati.
- **Source document:** `docs/course/Projects_List_2026.pdf` (the official list of potential projects).

The course's main theme is **AI and neuroscience**. Every part of this project should make the biological ↔ artificial link explicit: which neuroscience idea an analysis comes from, and what the result says back to neuroscience. See `docs/background/neuro_connections.md`.

## Project rules (from the PDF's notes)

1. Students may choose one of the listed projects.
2. Students may design their own project, which requires approval from the course instructors. A custom project:
   - should focus on the main theme of the class, i.e. AI and neuroscience;
   - cannot be the same as a project the student is pursuing for another class;
   - cannot be the same as a project the student is currently conducting in a lab;
   - should be roughly comparable in scope and difficulty to the listed ones.

**Our status:** we chose **Project 5** from the list, so the core project is pre-approved. Our two extensions (context-dependent regrouping; Jacobian lens) stay within Project 5's question. T-001 includes confirming with course staff that no extra approval is needed.

## Project 5, verbatim from the list

> **5. Can the geometry of internal LLM representations predict in-context generalization behavior?** (Difficulty: 5 / 10)
>
> Large language models are known to be effective at in-context learning, where they are prompted with example demonstrations of some task that does not appear in their pretraining data and are then asked to perform the task on new samples. One simple type of in-context learning task is few-shot categorization. In this paradigm, the model is prompted with pairs of existing concepts and the desired response for each. For example, you could prompt the model with "dog → A, cat → A, bird → A" and "tree → B, flower → B, fern → B" and then ask it "bat → ?". From the given examples, the model must infer that the synthetic categories foo and bar refer to animals and plants. This is an example of an easy task, but can you construct harder ones by using less clearly related examples.
>
> Given such a task, can you predict the model's accuracy based on the geometry of its internal representations of each of the concepts (dog, car, bird, tree, flower, fern, bat)? A recent mathematical theory (paper linked) shows how the geometry of the internal model representations affects few-shot classification performance, assuming a very simple learning rule, but it is unclear whether this simple model can be applied to predict the behavior of LLMs. Can you apply this theory to link model representations and behavior in a quantitative way?
>
> Reference: https://www.pnas.org/doi/10.1073/pnas.2200800119 (Sorscher, Ganguli & Sompolinsky 2022)

## How our project maps onto the brief

| Brief asks | Where we do it |
|---|---|
| Few-shot categorization tasks, including harder ones | Exp 1 category pairs with graded relatedness; Exp 2 crossed attributes (PLAN §4.2–4.3) |
| Predict accuracy from the geometry of concept representations | Exp 1: Sorscher theory on neutral representations (T-022–T-024) |
| Apply the PNAS theory quantitatively | Eq. 1 implemented, validated against Monte Carlo and the official code, compared with LLM error (T-022–T-024) |
| *(Extension)* Same items, different demonstrated grouping; representational shift across layers | Exp 2 (T-030–T-034) |
| *(Extension)* Jacobian-lens features vs full hidden states | Exp 3 (T-040–T-045) |

Adding the extensions raises the difficulty from the listed 5 to roughly 7. That's why the plan treats Exp 1 as the MVP that must land first (PLAN §8).

## Related projects on the same list

These are useful for methods and framing. We are not doing them.

- **Project 8:** do different AI models converge on the same representation? (CKA, RSA, mutual nearest neighbors). Relevant to our RSA/representation-comparison tooling.
- **Project 9:** how robust are intrinsic-dimension claims in LMs? Relevant to estimating participation ratio and to outlier dimensions.
- **Project 10:** decodable vs. used: does a linear probe find what the model relies on? Directly relevant to H2 (decodable ≠ used) and to the causal steering stretch goal (T-044).

## Compute hints from the list

Other LLM projects on the list specify "one 16 GB GPU; restrict to models of ≤3B parameters (or load larger ones in 8-bit)" and mention storing checkpoints in `/scratch`. This suggests a course cluster with scratch storage. **Confirm what applies to us in T-001.**

## Logistics (fill in via T-001)

| Item | Value |
|---|---|
| Final deadline | TBD |
| Intermediate milestones (proposal / check-in) | TBD |
| Report format & length | TBD |
| Presentation / poster | TBD |
| Code submission requirements | TBD |
| Team (solo / group, roles) | TBD |
| Compute available (cluster, GPU type, scratch quota) | TBD |
| Grading criteria | TBD |
| Instructor approval needed for extensions? | TBD (likely no; Project 5 is pre-approved) |
