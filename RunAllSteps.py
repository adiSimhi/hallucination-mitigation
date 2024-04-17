import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import scipy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Levenshtein import distance as lev
from InteventionByDetection import InterventionByDetection
from ModelInside import ModelInside
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

initial_dataset_path = "datasets/"

ending = ".json"


# ending = "one_shot_removing_threshold_on_prompt.json" # for new test on halluQA
def create_dataset(dataset_name, dataset_path, threshold, model_name="GOAT-AI/GOAT-7B-Community"):
    """
    create dataset
    :param dataset_name:
    :param dataset_path:
    :return:
    """
    # import
    if not os.path.exists(initial_dataset_path):
        print(f"{initial_dataset_path} does not exist")
        return
    else:
        print(f"{initial_dataset_path} exists")
    no_context = False
    if "no_context" in dataset_name:
        no_context = True
    if dataset_name == "disentQA" or "disentQA" in dataset_name or "no_context" in dataset_name:
        print(f"disentQA dataset creation")
        if no_context:
            print(f"no context addition {no_context}", flush=True)
            from DatasetCreationWithoutContext import CreateDataset
        else:

            from DatasetCreationFromDisentQA import CreateDataset
        # create dataset
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}",
                                         non_hall_save_path=f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}",
                                         general_save_path=f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}")
    # create dataset
    dataset_creation.save_data(data = dataset_creation.non_hall_dataset, path =f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}")
    dataset_creation.save_data(data = dataset_creation.hall_dataset, path =f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}")
    dataset_creation.save_data(data = dataset_creation.general_dataset, path =f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}")


def run_initial_test_on_dataset(threshold, model_name="GOAT-AI/GOAT-7B-Community", dataset_size=1000,
                                dataset_name="disentQA", concat_answer=False):
    print(
        f"threshold {threshold} model {model_name} dataset size {dataset_size} dataset name {dataset_name} concat_answer {concat_answer}")
    print(f"{initial_dataset_path=}")
    MLPCheck = ModelInside("results/",
                        f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}",
                        f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}",
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=concat_answer)
    all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.generate_data()
    import random
    random.seed(42)
    path_general = f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if not os.path.exists(f"{path_general.replace('.json', 'random_shuffle.json')}"):
        print(f"{path_general} does not exist")
        with open(path_general, "r") as f:
            data = json.load(f)
        random.shuffle(data)
        with open(f"{path_general.replace('.json', 'random_shuffle.json')}", "w") as f:
            json.dump(data, f)


def run_model_with_intervention_by_detection(threshold, model_name, dataset_size=1000, dataset_name="disentQA", alpha=5,
                                             threshold_acc=0.65, static_intervention=False,
                                             concat_answer=False, use_classifier_for_intervention=False,seed=None):
    print(
        f"threshold {threshold} model {model_name} dataset size {dataset_size} dataset name {dataset_name} alpha {alpha}seed {seed} use_classifier_for_intervention {use_classifier_for_intervention}")
    print(f"{initial_dataset_path=}")
    no_context_dataset = True if "no_context" in dataset_name else False
    static_intervention = static_intervention
    concatenate_answer = concat_answer
    on_test_set = True
    # seed = None
    addition_to_path = ""
    if on_test_set:
        addition_to_path = "_test_set"
    if use_classifier_for_intervention:
        addition_to_path = addition_to_path + "_classifier"
    if seed is not None:
        addition_to_path = addition_to_path + f"_seed{seed}"
    print(f"{static_intervention=} {concatenate_answer=}")
    intervene = InterventionByDetection("results/",
                                        f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending.replace('.json', 'random_shuffle.json')}",
                                        f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending.replace('.json', 'random_shuffle.json')}",
                                        general_data_path=f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}",
                                        model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                                        threshold_of_data=threshold, use_mlp=False, use_attention=False,
                                        use_heads=False, use_residual=False, alpha=alpha,
                                        static_intervention=static_intervention, concatenate_answer=concatenate_answer,
                                        on_test_set=on_test_set, seed_train_val=seed,
                                        use_classifier_for_intervention=use_classifier_for_intervention)
    intervene.intervention.min_acc = threshold_acc
    intervene.intervention.static_random_intervention = False
    options = [[True, False, False, False], [False, True, False, False], [False, False, True, False],
               [False, False, False, True], [False, False, False, False]]
    all_results_with = []
    all_results_without = []
    all_results_general = []
    for option in options:
        attn = option[0]
        mlp = option[1]
        heads = option[2]
        residual = option[3]
        print(f"{attn=} {mlp=} {heads=} {residual=}")
        intervene.set_what_to_intervene_on(use_attention=attn, use_mlp=mlp, use_heads=heads, use_residual=residual)
        print(f"{intervene.use_attention=} {intervene.use_mlp=} {intervene.use_heads=} {intervene.use_residual=}")
        assert intervene.use_attention == attn and intervene.use_mlp == mlp and intervene.use_heads == heads and intervene.use_residual == residual
        print(f"grounded")
        answer_without_rank, generated_answers_without, perplexity_without, generated_text_without,wikipedia_pp_without, wiki_generated_without = intervene.intervention.run_dataset_with_hook(
            intervene.test_without_examples, "without", check_mlp_out=intervene.mlp_output_of_test_without_examples,
            check_attention_out=intervene.attention_output_of_test_without_examples,
            check_residual_out=intervene.residual_output_of_test_without_examples,
            no_context_dataset=no_context_dataset, calculate_wiki_pp=True)
        print(f"hallucinate")

        answer_with_rank, generated_answers_with, perplexity_with, generated_text_with,  wikipedia_pp_with, wiki_generated_with = intervene.intervention.run_dataset_with_hook(
            intervene.test_with_examples, "with", check_mlp_out=intervene.mlp_output_of_test_with_examples,
            check_attention_out=intervene.attention_output_of_test_with_examples,
            check_residual_out=intervene.residual_output_of_test_with_examples, no_context_dataset=no_context_dataset,
            calculate_wiki_pp=False)
        print(f"general")
        answer_general_rank, generated_answers_general, perplexity_general, generated_text_general,  wikipedia_pp_general, wiki_generated_general = intervene.intervention.run_dataset_with_hook(
            intervene.test_general_examples, "general", no_context_dataset=no_context_dataset, calculate_wiki_pp=False)


        with_results = {"answer_with_rank": answer_with_rank,
                        "generated_answers_with": generated_answers_with, "perplexity_with": perplexity_with,
                        "generated_text_with": generated_text_with,
                        "wikipedia_pp_with": wikipedia_pp_with, "wiki_generated_with": wiki_generated_with}
        without_results = {"answer_without_rank": answer_without_rank,
                           "generated_answers_without": generated_answers_without,
                           "perplexity_without": perplexity_without, "generated_text_without": generated_text_without,
                           "wikipedia_pp_without": wikipedia_pp_without,
                           "wiki_generated_without": wiki_generated_without}
        general_results = { "answer_general_rank": answer_general_rank,
                           "generated_answers_general": generated_answers_general,
                           "perplexity_general": perplexity_general, "generated_text_general": generated_text_general,
                           "wikipedia_pp_general": wikipedia_pp_general,
                           "wiki_generated_general": wiki_generated_general}
        all_results_with.append(with_results)
        all_results_without.append(without_results)
        all_results_general.append(general_results)
        assert len(with_results.keys()) == len(without_results.keys()) == len(general_results.keys())
        # save the generated text to a json file
        path_with = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{attn}_mlp{mlp}_heads{heads}_residual{residual}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}with.json"
        if not os.path.exists(os.path.dirname(path_with)):
            os.makedirs(os.path.dirname(path_with))
        with open(path_with, 'w') as f:
            json.dump(with_results, f)
        path_without = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{attn}_mlp{mlp}_heads{heads}_residual{residual}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}without.json"
        if not os.path.exists(os.path.dirname(path_without)):
            os.makedirs(os.path.dirname(path_without))
        with open(path_without, 'w') as f:
            json.dump(without_results, f)
        # save the generated text to a json file
        path_general = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{attn}_mlp{mlp}_heads{heads}_residual{residual}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}general.json"
        if not os.path.exists(os.path.dirname(path_general)):
            os.makedirs(os.path.dirname(path_general))
        with open(path_general, 'w') as f:
            json.dump(general_results, f)

    dataset_without_test = intervene.test_without_examples
    dataset_with_test = intervene.test_with_examples
    dataset_general = intervene.test_general_examples
    full_with_dataset = intervene.hallucinations_examples
    full_without_dataset = intervene.non_hallucinations_examples
    full_general_dataset = intervene.general_examples
    del intervene
    path_with = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{False}_mlp{False}_heads{False}_residual{False}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}with.json"
    path_without = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{False}_mlp{False}_heads{False}_residual{False}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}without.json"
    path_general = "results/" + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold}/concat_answer{False}_size{dataset_size}/intervention_output/attn{False}_mlp{False}_heads{False}_residual{False}_alpha{alpha}_acc_threshold{threshold_acc}_concatenate_answer{concatenate_answer}_static{static_intervention}{addition_to_path}general.json"
    all_results_with, all_results_without, all_results_general = calc_perplexity(options, path_with, path_without,
                                                                                 path_general, dataset_with_test,
                                                                                 dataset_without_test, dataset_general)

    pretty_print_results(all_results_with=all_results_with, all_results_without=all_results_without,
                         all_results_general=all_results_general, threshold=threshold, model_name=model_name,
                         dataset_size=dataset_size, dataset_name=dataset_name, alpha=alpha,
                         options=options, full_with_dataset=full_with_dataset,
                         full_without_dataset=full_without_dataset, full_general_dataset=full_general_dataset)


def pretty_print_results(all_results_with, all_results_without, all_results_general, threshold, model_name,
                         dataset_size, dataset_name, alpha, options, full_with_dataset,
                         full_without_dataset, full_general_dataset):
    # pretty print all answers
    print(
        f"threshold {threshold} model {model_name} dataset size {dataset_size} dataset name {dataset_name} alpha {alpha} ")

    # final results
    print(f"final results hallucinate and grounded:")
    print(
        f"attn & mlp & heads & residual & acc classification & acc generation & perplexity & wikipedia perplexity \\\\\\midrule")
    acc_classification_average = []
    acc_generation_average = []
    for i, o in enumerate(options):
        assert len(all_results_without[i]['answer_without_rank']) == len(all_results_with[i][
                                                                             'answer_with_rank']), f"{len(all_results_without[i]['answer_without_rank'])=} {len(all_results_with[i]['answer_with_rank'])=}"
        acc_classification = round(((sum([1 for j in all_results_without[i]['answer_without_rank'] if j <= 0]) + sum(
            [1 for j in all_results_with[i]['answer_with_rank'] if j <= 0])) * 100) / (
                                           len(all_results_without[i]['answer_without_rank']) + len(
                                       all_results_with[i]['answer_with_rank'])), 2)
        acc_classification_average.append(acc_classification)
        acc_generation = round(((sum([1 for j in all_results_without[i]['generated_answers_without'] if j < 0]) + sum(
            [1 for j in all_results_with[i]['generated_answers_with'] if j < 0])) * 100) / (
                                       len(all_results_without[i]['generated_answers_without']) + len(
                                   all_results_with[i]['generated_answers_with'])), 2)
        acc_generation_average.append(acc_generation)
        prob_score = np.round(np.mean(all_results_without[i]['answer_without_prob']) + np.mean(
            all_results_with[i]['answer_with_prob']) / 2, 2)
        perplexity = np.round((np.mean(all_results_without[i]['perplexity_without']) + np.mean(
            all_results_with[i]['perplexity_with'])) / 2, 2)
        wikipedia_perplexity = np.round(np.mean(all_results_without[i]['wikipedia_pp_without']), 2)
        print(
            f"{o[0]} & {o[1]} & {o[2]} & {o[3]} & {acc_classification}  & {acc_generation} & {perplexity}  & {wikipedia_perplexity} \\\\")
    print(
        f"average acc classification {np.mean(acc_classification_average)} average acc generation {np.mean(acc_generation_average)}")

    if all_results_general[0]["answer_general_rank"] is not None:
        print(f"final results with wighted average:")
        hall_count = len(full_with_dataset)
        non_hall_count = len(full_without_dataset)
        general_count = len(full_general_dataset)
        print(
            f"attn & mlp & heads & residual & acc classification & acc generation  & perplexity &  wikipedia perplexity\\\\\\midrule")
        for i, o in enumerate(options):
            classification_acc = round(((sum([1 for j in all_results_general[i]['answer_general_rank'] if j <= 0]) * (
                    general_count / len(all_results_general[i]['answer_general_rank'])) + sum(
                [1 for j in all_results_with[i]['answer_with_rank'] if j <= 0]) * (
                                                 hall_count / len(all_results_with[i]['answer_with_rank'])) + sum(
                [1 for j in all_results_without[i]['answer_without_rank'] if j <= 0]) * (
                                                 non_hall_count / len(
                                             all_results_without[i]['answer_without_rank']))) * 100) / (
                                               general_count + hall_count + non_hall_count), 2)
            prob_score = np.round((np.mean(all_results_without[i]['answer_without_prob']) * non_hall_count + np.mean(
                all_results_with[i]['answer_with_prob']) * hall_count + np.mean(
                all_results_general[i]['answer_general_prob']) * general_count) / (
                                          general_count + hall_count + non_hall_count), 2)
            generation_acc = round(((sum([1 for j in all_results_general[i]['generated_answers_general'] if j < 0]) * (
                    general_count / len(all_results_general[i]['answer_general_rank'])) + sum(
                [1 for j in all_results_with[i]['generated_answers_with'] if j < 0]) * (
                                             hall_count / len(all_results_with[i]['answer_with_rank'])) + sum(
                [1 for j in all_results_without[i]['generated_answers_without'] if j < 0]) * (
                                             non_hall_count / len(
                                         all_results_without[i]['answer_without_rank']))) * 100) / (
                                           general_count + hall_count + non_hall_count), 2)
            perplexity = np.round((np.mean(all_results_without[i]['perplexity_without']) * non_hall_count + np.mean(
                all_results_with[i]['perplexity_with']) * hall_count + np.mean(
                all_results_general[i]['perplexity_general']) * general_count) / (
                                          general_count + hall_count + non_hall_count), 4)
            wikipedia_perplexity = np.round(np.mean(all_results_without[i]["wikipedia_pp_without"]), 2)
            print(
                f"{o[0]} & {o[1]} & {o[2]} & {o[3]} & {classification_acc}  & {generation_acc} & {perplexity}  & {wikipedia_perplexity}  \\\\")


def calc_perplexity(options, path_with, path_without, path_general, dataset_with_test, dataset_without_test,
                    dataset_general):
    all_results_with = []
    all_results_without = []
    all_results_general = []

    for option in options:
        # create and save perplexity

        attn = option[0]
        mlp = option[1]
        heads = option[2]
        residual = option[3]
        paraphraze_prompt_index = 5
        print(f"{attn=} {mlp=} {heads=} {residual=}")
        # load the generated text
        curr_path_with = path_with.replace(f"attn{False}_mlp{False}_heads{False}_residual{False}",
                                           f"attn{attn}_mlp{mlp}_heads{heads}_residual{residual}")
        with open(curr_path_with, 'r') as f:
            with_results = json.load(f)
        perplexity_with = perplexity_score(dataset=dataset_with_test,
                                           generated_text=with_results["generated_text_with"],
                                           model_name="mistralai/Mistral-7B-v0.1",
                                           paraphraze_prompt_index=paraphraze_prompt_index)
        with_results["perplexity_with"] = perplexity_with
        with open(curr_path_with, 'w') as f:
            json.dump(with_results, f)
        all_results_with.append(with_results)
        curr_path_without = path_without.replace(f"attn{False}_mlp{False}_heads{False}_residual{False}",
                                                 f"attn{attn}_mlp{mlp}_heads{heads}_residual{residual}")
        with open(curr_path_without, 'r') as f:
            without_results = json.load(f)
        perplexity_without = perplexity_score(dataset=dataset_without_test,
                                              generated_text=without_results["generated_text_without"],
                                              model_name="mistralai/Mistral-7B-v0.1",
                                              paraphraze_prompt_index=paraphraze_prompt_index)
        without_results["perplexity_without"] = perplexity_without
        with open(curr_path_without, 'w') as f:
            json.dump(without_results, f)
        all_results_without.append(without_results)
        curr_path_general = path_general.replace(f"attn{False}_mlp{False}_heads{False}_residual{False}",
                                                 f"attn{attn}_mlp{mlp}_heads{heads}_residual{residual}")
        with open(curr_path_general, 'r') as f:
            general_results = json.load(f)
        perplexity_general = perplexity_score(dataset=dataset_general,
                                              generated_text=general_results["generated_text_general"],
                                              model_name="mistralai/Mistral-7B-v0.1",
                                              paraphraze_prompt_index=paraphraze_prompt_index)
        general_results["perplexity_general"] = perplexity_general
        with open(curr_path_general, 'w') as f:
            json.dump(general_results, f)
        all_results_general.append(general_results)
    assert len(all_results_with) == len(all_results_without) == len(all_results_general) == len(options)
    return all_results_with, all_results_without, all_results_general


def perplexity_score(dataset, model_name, generated_text, paraphraze_prompt_index):
    print(f"start perplexity score", flush=True)
    gptj_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map="auto")
    gptj_model.eval()

    gptj_tok = AutoTokenizer.from_pretrained(model_name)
    perplexity = []
    for index, point in enumerate(dataset):

        generated = generated_text[index][0]
        praphraze_prompt = point[paraphraze_prompt_index]
        perplexity.append(fluency_score(praphraze_prompt, generated, gptj_model, gptj_tok))
        if perplexity[-1] == 0:
            if perplexity[-1] == 0 and len(generated) > 0:
                print(f"perplexity is 0 for non zero generation {generated=}")
            perplexity = perplexity[:-1]
    del gptj_model
    del gptj_tok
    torch.cuda.empty_cache()
    return perplexity


def fluency_score(initial_text, generated_text, gptj_model, gptj_tok):
    """
    get the fluency score of the text by calculating the perplexity to generate this text using the model
    :param text:
    :return:
    """
    # print(f"{initial_text=} {generated_text=}")
    perplexity_arr = []
    input_ids = gptj_tok.encode(initial_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gptj_model(input_ids, labels=input_ids.clone())
        loss_pre = outputs.loss

    if len(input_ids[0]) > 1000:
        print(f"long context for pp {len(input_ids[0])=}")
        return 0
    input_ids_full = gptj_tok.encode(generated_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # run model using cross entropy loss sum loss
        outputs = gptj_model(input_ids_full, labels=input_ids_full.clone())
        loss = outputs.loss

    print(f"{loss.item()=} {loss_pre.item()=} {len(input_ids_full[0])=} {len(input_ids[0])=}")
    if len(input_ids[0]) + 60 < len(input_ids_full[0]):
        print(f"long generation for pp {len(input_ids_full[0])=} {generated_text=}")
    assert len(input_ids_full[0]) >= len(input_ids[0]), f"{len(input_ids[0])=} {len(input_ids_full[0])=}"
    pp = (loss * len(input_ids_full[0]) - loss_pre * len(input_ids[0])) / (
            len(input_ids_full[0]) - len(input_ids[0]))
    # print(f"{pp=} {torch.exp(pp)=}")
    pp = torch.exp(pp)

    if len(input_ids_full[0]) == len(input_ids[0]):
        pp = torch.tensor(0)
    if pp.item() == float("inf"):  # for not want to use inf
        print(f"pp is inf {pp=}")
        pp = torch.tensor(0)
    return pp.item()


if __name__ == "__main__":
    print(f"git version {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()}")
    print(f"git diff {subprocess.check_output(['git', 'diff']).decode('ascii').strip()}")
    print(f"start time {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    # create yml file of the environment with the date in the name
    os.makedirs("environment/", exist_ok=True)
    os.system(f"conda env export > environment/environment_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.yml")
    os.makedirs("results/", exist_ok=True)
    parser = argparse.ArgumentParser()
    # dataset size
    parser.add_argument("--dataset_size", type=int, default=1000)
    # model name
    parser.add_argument("--model_name", type=str, default="GOAT-AI/GOAT-7B-Community")
    # threshold
    parser.add_argument("--threshold", type=float, default=1)
    # dataset name
    parser.add_argument("--dataset_name", type=str, default="disentQA")
    # threshold for intervention by detection acc
    parser.add_argument("--threshold_acc", type=float, default=0.65)
    # dataset path
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/v10-simplified_simplified-nq-train_factual_counterfactual_disentangled_baseline_train_split.csv")
    # run dataset creation
    parser.add_argument("--run_dataset_creation", type=bool, default=False,
                        help="run dataset creation - create the hallucination and non-hallucination datasets")
    parser.add_argument("--run_initial_test", type=bool, default=False, help="run initial test on the dataset and create the info for that")

    parser.add_argument("--alpha", type=float, default=5, help="alpha for the magnitude of the intervention")
    parser.add_argument("--static_intervention", type=bool, default=False, help="If true static intervention else dynamic intervention")
    parser.add_argument("--concat_answer", type=bool, default=False, help="If true d-compute post-answer else pre-answer")
    parser.add_argument("--use_classifier_for_intervention", type=bool, default=False,
                        help="use classifier for intervention instead of mass-mean")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--InterventionByDetection", type=bool, default=False, help="Intervention By Detection")
    print(f"{parser.parse_args().alpha=}", flush=True)
    if parser.parse_args().run_dataset_creation:
        # create dataset
        create_dataset(parser.parse_args().dataset_name, parser.parse_args().dataset_path,
                       parser.parse_args().threshold, parser.parse_args().model_name)
    if parser.parse_args().run_initial_test:
        # run initial test
        run_initial_test_on_dataset(threshold=parser.parse_args().threshold, model_name=parser.parse_args().model_name,
                                    dataset_size=parser.parse_args().dataset_size,
                                    dataset_name=parser.parse_args().dataset_name,
                                    concat_answer=parser.parse_args().concat_answer)
    # intervention and detection
    if parser.parse_args().InterventionByDetection:
        run_model_with_intervention_by_detection(threshold=parser.parse_args().threshold,
                                                 model_name=parser.parse_args().model_name,
                                                 dataset_size=parser.parse_args().dataset_size,
                                                 dataset_name=parser.parse_args().dataset_name,
                                                 alpha=parser.parse_args().alpha,
                                                 threshold_acc=parser.parse_args().threshold_acc,
                                                 static_intervention=parser.parse_args().static_intervention,
                                                 concat_answer=parser.parse_args().concat_answer,
                                                 use_classifier_for_intervention=parser.parse_args().use_classifier_for_intervention,
                                                 seed=parser.parse_args().seed)
