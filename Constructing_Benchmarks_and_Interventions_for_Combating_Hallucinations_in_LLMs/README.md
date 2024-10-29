# Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs

Paper link: [Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs](https://arxiv.org/abs/2404.09971)

## Abstract
Large language models (LLMs) are susceptible to hallucination, which sparked a widespread effort to detect and prevent them. Recent work attempts to mitigate hallucinations by intervening in the model's computation during generation, using different setups and heuristics. Those works lack separation between different hallucination causes. In this work, we first introduce an approach for constructing datasets based on the model knowledge for detection and intervention methods in closed-book and open-book question-answering settings. We then characterize the effect of different choices for intervention, such as the intervened components (MLPs, attention block, residual stream, and specific heads), and how often and how strongly to intervene. We find that intervention success varies depending on the component, with some components being detrimental to language modeling capabilities. Finally, we find that interventions can benefit from pre-hallucination steering direction instead of post-hallucination. 

## Final datasets
The final datasets for both Llama2-7B and Goat-7B of both the open-book and closed-book settings can be found in the datasets folder.

## How to run the code?
This code works for Linux.
First create and activate the environment specified by `environemnt.yml`

### Create the dataset

If you plan to use the disentQA dataset, you need to first download it from the
following [link](https://docs.google.com/document/d/1Z4vA7ifMQTk5YBF3BEYCFSnIvXCYaznLP_7VBcXPEeU/edit#heading=h.3prdi119z3tn)
and load a download a file named
v10-simplified_simplified-nq-train_factual_counterfactual_disentangled_baseline_train_split.csv. Than save it at the
datasets folder.

To create the labeled dataset and save them in the datasets folder run the following command:

```bash
python RunAllSteps.py --dataset_size 1000 --model_name model_name  --threshold 1 
--dataset_name disentQA/trivia_qa_no_context --run_dataset_creation True  
```

After this step you will have three files in the datasets folder: one that starts with Hallucinate, one that starts with
Non-Hallucinate and one that starts with General. The Hallucinate file contains the datapoints that are classified as
hallucinations, the Non-Hallucinate file contains the datapoints that are classified as non-hallucinations (grounded)
and the General file contains the datapoints that are not classified to either label.

Each example in the dataset of the open-book setting is a tuple of the form: (prompt_without_context, parametric_answer, contextual_answer,
parametric_answer_tokens, contextual_answer_tokens, prompt_with_context,
                         count_know_out_of_5, rank_of_the_contextual_answer - rank_of_the_parametric_answer) 

and for the closed-book setting:  (clean_prompt, golden_answer, golden_answer, golden_answer_token, golden_answer_token,
                         prompt_with_bad_shots, count_know_out_of_5,
                         rank_with_bad_shot - rank_with_good_shot
                         )

### Create Inner States Information

To create inner states information for dataset_name (disentQA/trivia_qa_no_context) run the following command (the inner
states files will be saved in folder
```results/model_name/dataset_name/threshold/concat_answer{False/True}_size{dataset_size}```):

```bash
python RunAllSteps.py --dataset_size 1000 --model_name model_name  --threshold 1 --dataset_name dataset_name --run_initial_test True
```

### Detection and Intervention

In this step other than the detection graphs you will also have a files of the intervention results for each different
configuration a unique file with the associate name in an inner
directory  ```results/model_name/dataset_name/threshold/concat_answer{False/True}_size{dataset_size}/intervention_output```

For example
```attnTrue_mlpFalse_headsFalse_residualFalse_alpha5.0_acc_threshold0.65_concatenate_answerFalse_staticTrue_test_set_seed200with.json```

Is a file with the intervention results for the configuration of attention=True, mlp=False, heads=False, residual=False,
alpha=5.0, acc_threshold(=TD)=0.65, concatenate_answer=False (pre-answer), static=True (static instead of dynamic
intervention), on test set (test_set), seed=200 and with.json is the file with the hallucinate dataset (
with=hallucinate,without=grounded,general=else-datapoints that are not classified to ether label).

To run detection and intervention run the following command

```bash
python RunAllSteps.py --dataset_size 1000 --model_name model_name  --threshold 1
--dataset_name disentQA/trivia_qa_no_context --InterventionByDetection True --alpha 5 --threshold_acc 0.65 --static_intervention True --seed 200
```
