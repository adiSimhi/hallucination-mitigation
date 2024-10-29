# Distinguishing Ignorance from Error in LLM Hallucinations

# Abstract

Large language models (LLMs) are susceptible to hallucinations---outputs that are ungrounded, factually incorrect, or inconsistent with prior generations. We focus on close-book Question Answering (CBQA), where previous work has not fully addressed the distinction between two possible kinds of hallucinations, namely, whether the model (1) does not hold the correct answer in its parameters or (2) answers incorrectly despite having the required knowledge. 
We argue that distinguishing these cases is crucial for detecting and mitigating hallucinations.
Specifically, case (2) may be mitigated by intervening in the model’s internal computation, as the knowledge resides within the model's parameters. 
In contrast, in case (1) there is no parametric knowledge to leverage for mitigation, 
so it should be addressed by resorting to an external knowledge source or abstaining.
To help distinguish between the two cases, we introduce Wrong Answer despite having Correct Knowledge (WACK), an approach for constructing model-specific datasets for the second hallucination type. 
Our probing experiments indicate that the two kinds of hallucinations are represented differently in the model's inner states. Next, 
 we show that datasets constructed using WACK exhibit variations across models, demonstrating that even when models share knowledge of certain facts, they still vary in the specific examples that lead to hallucinations.
 Finally, we show that training a probe on our WACK datasets leads to better hallucination detection of case (2) hallucinations than using the common generic one-size-fits-all datasets. 

## Final datasets

The final datasets for the three models (Mistral-7B-v0.3, Llama-3.1-8B and Gemma-2-9B) on the two datasets (Natural Quetions and TriviaQA) can be found in datasets folder.

## How to run the code?

This code works for Linux.
First create and activate the environment specified by `environemnt.yml`

### Create the dataset

To create the labeled dataset and save them in the datasets folder run the following command:

```bash
python RunAllSteps.py --dataset_size 1000 --model_name model_name  --threshold 1 
--dataset_name natural_question_no_context/trivia_qa_no_context --run_dataset_creation True  
```

After this step you will have three files in the datasets folder: one that starts with Hallucinate, one that starts with
Non-Hallucinate and one that starts with General. The Hallucinate file contains the datapoints that are classified as
hallucinations, the Non-Hallucinate file contains the datapoints that are classified as non-hallucinations (factually-correct),
and the General file contains the datapoints that are ether hallucination because of not-knowing (0 in count_knowledge)
or are from a different knowledge place in the spectrum.

We can create the same way dataset using Alice setting, and create files that their name starts with Alice, by adding
the flag `--alice True` to the command.

Additionally, this step also creates the Generic (static) dataset that contains only hallucinations and grounded
data points with no regard to the two hallucinations and to the model's specific hallucinations.

Each data point in the specific dataset contains:
clean_prompt, golden_answer, wrong_answer, golden_answer_token, wrong_answer_token, prompt_with_bad_shots/alice, count_knowledge,-1

For the generic dataset each data point contains:
clean_prompt, golden_answer, wrong_answer, golden_answer_token, wrong_answer_token, prompt_with_bad_shots/alice, -1, -1

### Create Inner States Information

To create inner states information for dataset_name (disentQA/trivia_qa_no_context) run the following command (the inner
states files will be saved in folder
```results/model_name/dataset_name/{threshold}/concat_answer{False/True}_size{dataset_size}```):

```bash
python RunAllSteps.py --dataset_size 1000 --model_name model_name  --threshold 1 --dataset_name dataset_name --run_initial_test True
```

### plot the results

To create the final results on bad-shot setting use the following command:

```bash

python RunAllSteps.py python RunAllSteps.py --plot_results True --know_hall_vs_do_not_know_hall_vs_know True
```

This will create the results for the accuracy of differentiating the two hallucinations and knowledge and save them in
the results folder.

Similarly to run the other results we can call with the different parameters. The parameters are:
--post_answer True/False to plot the results of generic vs specific dataset on the regular (post-answer) setting
--pre_answer True/False to plot the results of generic vs specific dataset on the pre-answer setting
--alice_vs_bad_shot True/False to plot the results of Alice vs bad-shot setting
