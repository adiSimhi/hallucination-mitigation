# Hallucination Mitigation

This repository includes code implementations for two research papers that explore hallucination detection and
mitigation in large language models (LLMs), focusing on different aspects of the problem.
"Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs" and "Distinguishing Ignorance from
Error in LLM Hallucinations"



## Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs

[README for this paper](Constructing_Benchmarks_and_Interventions_for_Combating_Hallucinations_in_LLMs/README.md)

The first paper contain an overview of intervention for Hallucination Mitigation: develop WACK benchmarks to test
different intervention methods, focusing on factors like the components being adjusted, and the frequency and intensity
of intervention. Revealing that
attention-based interventions
outperform the residual stream. And introduce Dynamic interventions
to apply mitigation only when necessary, offering a more robust approach.

## Distinguishing Ignorance from Error in LLM Hallucinations

[README for this paper](Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations/README.md)

The second paper addresses two types of hallucinations in LLMs during closed-book question
answering (CBQA): (1) when the model lacks the correct answer in its parameters, and (2) when the model answers
incorrectly despite possessing the necessary knowledge. We argue that distinguishing these cases is essential,
as each requires different mitigation strategies. To assist in differentiating these cases, we introduce the WACK
dataset (Wrong Answer despite
having Correct Knowledge), which is model-specific. Probing experiments reveal distinct internal
representations for each hallucination type, with WACK datasets showing variation in the model-specific-datasets between models. This model-specific
approach enhances detection of case (2) hallucinations compared to using generic datasets.













