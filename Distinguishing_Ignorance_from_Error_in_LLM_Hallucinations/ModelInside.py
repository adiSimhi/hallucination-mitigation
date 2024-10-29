"""
In this file we will save the inner states of the model for the dataset with and without hallucinations
"""

import functools
import gc
import json
import os
import random
import subprocess
from typing import List

import numpy as np
import psutil as psutil
import torch

from InfoModelUsingWrapper import InnerStatesUsingWrapper
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelInside():

    def __init__(self, path_to_save_results, data_path_without_hallucinations, data_path_with_hallucinations,
                 model_name="GOAT-AI/GOAT-7B-Community", dataset_size=1000, dataset_name="disentQA", threshold_of_data=None,
                 concat_answer=False,static_dataset=False, alice=False):
        self.path_to_save_results = path_to_save_results
        self.MODEL_NAME = model_name
        self.dataset_size = dataset_size
        self.dataset_name = dataset_name
        self.prob_low_threshold_of_answer_without_context = -1
        self.prob_high_threshold_of_answer_without_context = 1
        self.concat_answer = concat_answer
        self.data_path_without_hallucinations = data_path_without_hallucinations
        self.data_path_with_hallucinations = data_path_with_hallucinations
        self.path_to_save_results = self.path_to_save_results + f"{self.MODEL_NAME.replace('/', '_')}" + f'{"/"}' + f"{self.dataset_name}/{threshold_of_data}/concat_answer{self.concat_answer}_size{self.dataset_size}/"
        if concat_answer and not static_dataset:
            self.path_to_save_results = self.path_to_save_results + "non_static/"
        if alice or (self.data_path_with_hallucinations!= None and "alice" in self.data_path_with_hallucinations.lower()):
            self.path_to_save_results = self.path_to_save_results + "alice/"
        print(self.path_to_save_results)
        # create directory to save results
        # mkdir to also work for os permission denied
        self.static_dataset = static_dataset
        os.makedirs(self.path_to_save_results, exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()

    def generate_data(self):
        """
        generate the data for all model's inner states and save it
        :return:
        """
        generate_text_func = self.get_inner_model_info_using_context_dataset if (
                    self.dataset_name == "disentQA") else self.get_inner_model_info_no_context_dataset
        self.data_with_hallucinations = self.load_dataset(self.data_path_with_hallucinations)
        random.seed(42)
        norm_mlp_with, norm_attn_with, all_mlp_vector_with_hall, all_attention_vector_with_all, heads_vectors_with, all_residual_vectors_with_hall = generate_text_func(
            self.data_with_hallucinations, tag='with_hallucinations', concatenate_answer=self.concat_answer,static=self.static_dataset)

        self.data_without_hallucinations = self.load_dataset(self.data_path_without_hallucinations)

        random.seed(42)

        norm_mlp_without, norm_attn_without, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_without, all_residual_vectors_without_hall = generate_text_func(
            self.data_without_hallucinations, tag='without_hallucinations', concatenate_answer=self.concat_answer)
        # randomly split the all_mlp_vector_with_hall to two parts
        # save the data - all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall
        self.save_all_data(all_mlp_vector_with_hall=all_mlp_vector_with_hall,
                           all_attention_vector_with_all=all_attention_vector_with_all,
                           all_mlp_vector_without_hall=all_mlp_vector_without_hall,
                           all_attention_vector_without_hall=all_attention_vector_without_hall,
                           heads_vectors_with=heads_vectors_with, heads_vectors_without=heads_vectors_without,
                           norm_mlp_with=norm_mlp_with,
                           norm_mlp_without=norm_mlp_without, norm_attn_with=norm_attn_with,
                           norm_attn_without=norm_attn_without,
                           all_residual_vectors_with_hall=all_residual_vectors_with_hall,
                           all_residual_vectors_without_hall=all_residual_vectors_without_hall)
        return all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_vectors_with_hall, all_residual_vectors_without_hall


    def generate_data_type_1(self,data_path):

        type_1_data = self.load_dataset(data_path)
        type_1_data = [d for d in type_1_data if d[-2] == 0]
        print(f"number of examples of type 1 is {len(type_1_data)}")
        norm_mlp_type1, norm_attn_type1, all_mlp_vector_type1, all_attention_vector_type1, heads_vectors_type1, all_residual_vectors_type1 = self.get_inner_model_info_no_context_dataset(
            type_1_data, tag='else_hallucinations', concatenate_answer=self.concat_answer,
            static=False)
        self.save_data(all_mlp_vector_type1, "all_mlp_vector_type1")
        self.save_data(all_attention_vector_type1, "all_attention_vector_type1")
        self.save_data(heads_vectors_type1, "heads_vectors_type1")
        self.save_data(all_residual_vectors_type1, "all_residual_vectors_type1")
        return all_mlp_vector_type1, all_attention_vector_type1, heads_vectors_type1, all_residual_vectors_type1


    def get_type_1_data(self,path):
        if os.path.exists(f"{self.path_to_save_results}_all_mlp_vector_type1.npy"):
            all_mlp_vector_type1 = np.load(f"{self.path_to_save_results}_all_mlp_vector_type1.npy")
            all_attention_vector_type1 = np.load(f"{self.path_to_save_results}_all_attention_vector_type1.npy")
            heads_vectors_type1 = np.load(f"{self.path_to_save_results}_heads_vectors_type1.npy")
            all_residual_vectors_type1 = np.load(f"{self.path_to_save_results}_all_residual_vectors_type1.npy")
        else:
            all_mlp_vector_type1, all_attention_vector_type1, heads_vectors_type1, all_residual_vectors_type1 = self.generate_data_type_1(
                path)
        return all_mlp_vector_type1, all_attention_vector_type1, heads_vectors_type1, all_residual_vectors_type1






    def save_data(self, data, name):
        """
        save the data to numpy file"""
        print(f"saving {name} data of type {type(data)}")
        np.save(f"{self.path_to_save_results}_{name}", data)

    def save_all_data(self, all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall,
                      all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, norm_mlp_with,
                      norm_mlp_without, norm_attn_with, norm_attn_without, all_residual_vectors_with_hall,
                      all_residual_vectors_without_hall):
        self.save_data(all_mlp_vector_with_hall, "all_mlp_vector_with_hall")
        self.save_data(all_attention_vector_with_all, "all_attention_vector_with_all")
        self.save_data(all_mlp_vector_without_hall, "all_mlp_vector_without_hall")
        self.save_data(all_attention_vector_without_hall, "all_attention_vector_without_hall")
        self.save_data(heads_vectors_with, "heads_vectors_with_no_projection")
        self.save_data(heads_vectors_without, "heads_vectors_without_no_projection")
        self.save_data(all_residual_vectors_with_hall, "all_residual_vectors_with_hall")
        self.save_data(all_residual_vectors_without_hall, "all_residual_vectors_without_hall")

    def load_all_data(self):
        names = ["all_mlp_vector_with_hall", "all_attention_vector_with_all", "all_mlp_vector_without_hall",
                 "all_attention_vector_without_hall", "heads_vectors_with_no_projection",
                 "heads_vectors_without_no_projection", "all_residual_vectors_with_hall",
                 "all_residual_vectors_without_hall"]
        datas = self.load_data(names)
        return datas[0], datas[1], datas[2], datas[3], datas[4], datas[5], datas[6], datas[7]

    def load_data(self, names):
        datas = []
        for name in names:
            data = np.load(f"{self.path_to_save_results}_{name}.npy")
            datas.append(data)
            # print data name,hash,last modification date
            print(f"{name} {os.path.getmtime(f'{self.path_to_save_results}_{name}.npy')}")
            print(f"hash data {subprocess.check_output(['sha1sum', f'{self.path_to_save_results}_{name}.npy'])}")
        return datas

    def load_dataset(self, data_path):
        """
        load the dataset
        :param data_path:
        :return: dataset
        """
        with open(data_path) as f:
            data = json.load(f)
        print(f"dataset size is {len(data)}")
        return data


    def get_inner_model_info_using_context_dataset(self, dataset, tag, concatenate_answer=False):
        """
        get the inner model information using the context dataset
        """
        all_mlp_results = []
        all_attention_results = []
        all_last_token_mlp_vectors = []
        all_last_token_attention_vectors = []
        heads_vectors_all_layers_all_examples = []
        all_last_token_residual_stream = []
        norm_calculator = InnerStatesUsingWrapper(MODEL_NAME=self.MODEL_NAME)
        prompt_index = 0
        parametric_answer = 1
        contextual_answer = 2
        if concatenate_answer:
            if "without" in tag:
                print("concatenate contextual answer")
                answer = contextual_answer
            else:
                print("concatenate parametric answer")
                answer = parametric_answer
        else:
            print("not concatenate answer")
            answer = ""
        paraphraze_prompt_index = -3
        logits_on_true_answer_without_context_index = -1
        number_of_examples_used = 0
        for index, point in enumerate(dataset):
            if index % 10 == 0:
                print(f"index is {index} with {number_of_examples_used=}", flush=True)
            if number_of_examples_used >= self.dataset_size:
                break

            if "without" in tag:
                assert point[-1] < 0
            else:
                assert point[-1] > 0
            number_of_examples_used += 1

            prompt = point[prompt_index]
            old_target = point[1]
            new_target = point[2]
            if answer != "":
                print("concatenate answer!!!")
                praphraze_prompt = point[paraphraze_prompt_index] + point[answer]
                if index < 5:
                    print(f"With concatenation prompt {praphraze_prompt=}")
            else:
                praphraze_prompt = point[paraphraze_prompt_index]
            # print(f"{prompt=} {old_target=} {new_target=} {praphraze_prompt=}")
            # prompt, old_target, new_target, old_token, new_token, praphraze_prompt, final_logits,logits_on_true_answer_without_context

            mlp_norm, attantion_norm, last_token_mlp_all_layers, last_token_attention_all_layers, heads_vectors, last_token_residual_stream = norm_calculator.generate_interactive(
                prompt=prompt, paraphraze_prompt=praphraze_prompt)
            all_mlp_results.append(mlp_norm)
            all_attention_results.append(attantion_norm)
            all_last_token_mlp_vectors.append(last_token_mlp_all_layers)
            all_last_token_attention_vectors.append(last_token_attention_all_layers)
            heads_vectors_all_layers_all_examples.append(heads_vectors)
            all_last_token_residual_stream.append(last_token_residual_stream)
        del norm_calculator
        gc.collect()
        torch.cuda.empty_cache()
        assert number_of_examples_used == index, f"{number_of_examples_used=} {index=}"
        assert len(all_mlp_results) == len(all_attention_results) == len(all_last_token_mlp_vectors) == len(
            all_last_token_attention_vectors) == len(heads_vectors_all_layers_all_examples) == len(
            all_last_token_residual_stream) == number_of_examples_used
        print(f"{number_of_examples_used=} using {index=} examples")
        return all_mlp_results, all_attention_results, np.array(all_last_token_mlp_vectors), np.array(
            all_last_token_attention_vectors), np.array(heads_vectors_all_layers_all_examples), np.array(
            all_last_token_residual_stream)

    def get_inner_model_info_no_context_dataset(self, dataset, tag, concatenate_answer=False,static=False):
        """
        get inner model info using no-context dataset
        """
        all_mlp_results = []
        all_attention_results = []
        all_last_token_mlp_vectors = []
        all_last_token_attention_vectors = []
        heads_vectors_all_layers_all_examples = []
        all_last_token_residual_stream = []
        norm_calculator = InnerStatesUsingWrapper(MODEL_NAME=self.MODEL_NAME)
        prompt_index = 0
        paraphraze_prompt_index = -3
        logits_on_true_answer_without_context_index = -1
        number_of_examples_used = 0
        parametric_answer = 1
        contextual_answer = 2

        for index, point in enumerate(dataset):
            if index % 10 == 0:
                print(f"index is {index} with {number_of_examples_used=}", flush=True)
            if number_of_examples_used >= self.dataset_size:
                break
            # use only examples with high confidence on the true answer given without context prompt
            number_of_examples_used += 1

            prompt = point[paraphraze_prompt_index]
            praphraze_prompt = point[
                paraphraze_prompt_index]  # prompt and paraphraze prompt are the same as we do not have context
            assert prompt == praphraze_prompt
            if concatenate_answer:
                if "without" in tag:
                    print("concatenate parametric answer")
                    answer = point[parametric_answer]
                    if not static:
                        answer = self.greedy_generation_to_generate_answer(norm_calculator.tok, norm_calculator.model,
                                                                           point[paraphraze_prompt_index],
                                                                           length=5)
                        print(
                            f"generated answer is {answer=} for {point[paraphraze_prompt_index]=} {point[contextual_answer]=} {point[parametric_answer]=}")
                else:
                    print("concatenate new answer")
                    answer = point[contextual_answer]
                    if not static:
                        answer = self.greedy_generation_to_generate_answer(norm_calculator.tok,norm_calculator.model, point[paraphraze_prompt_index],
                                                                           length=5)
                        print(f"generated answer is {answer=} for {point[paraphraze_prompt_index]=} {point[contextual_answer]=} {point[parametric_answer]=}")
            else:
                print("not concatenate answer")
                answer = ""
            if answer != "":
                print("concatenate answer!!!")
                praphraze_prompt = point[paraphraze_prompt_index] + answer
                if index < 5:
                    print(f"With concatenation prompt {praphraze_prompt=}")
            else:
                praphraze_prompt = point[paraphraze_prompt_index]
            # print(f"{prompt=} {old_target=} {new_target=} {praphraze_prompt=}")
            # prompt, old_target, new_target, old_token, new_token, praphraze_prompt, final_logits,logits_on_true_answer_without_context

            mlp_norm, attantion_norm, last_token_mlp_all_layers, last_token_attention_all_layers, heads_vectors, last_token_residual_stream = norm_calculator.generate_interactive(
                prompt=prompt, paraphraze_prompt=praphraze_prompt)
            all_mlp_results.append(mlp_norm)
            all_attention_results.append(attantion_norm)
            all_last_token_mlp_vectors.append(last_token_mlp_all_layers)
            all_last_token_attention_vectors.append(last_token_attention_all_layers)
            heads_vectors_all_layers_all_examples.append(heads_vectors)
            all_last_token_residual_stream.append(last_token_residual_stream)
        del norm_calculator
        gc.collect()
        torch.cuda.empty_cache()
        # assert number_of_examples_used == index or number_of_examples_used == self.dataset_size, f"{number_of_examples_used=} {index=}"
        print(f"{number_of_examples_used=} using {index=} examples")
        return all_mlp_results, all_attention_results, np.array(all_last_token_mlp_vectors), np.array(
            all_last_token_attention_vectors), np.array(heads_vectors_all_layers_all_examples), np.array(
            all_last_token_residual_stream)


    def greedy_generation(self,tok, model, prompt, length=5):
        input_ids = \
            tok(prompt, padding=True, return_token_type_ids=False, return_tensors="pt")[
                "input_ids"].to(device)
        with torch.no_grad():
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + length), do_sample=False,
                                       pad_token_id=tok.eos_token_id, num_beams=1, top_p=None, temperature=None,
                                       attention_mask=torch.ones_like(input_ids))
        # only new generated tokens
        generated = tok.decode(model_out[0], skip_special_tokens=True)[len(prompt):]
        return generated

    def greedy_generation_to_generate_answer(self,tok, model, prompt, length=5):
        generated = self.greedy_generation(tok,model, prompt, length)
        generated = generated.split("question:")[0].split("question")[0].replace("\n","").replace("Incorrect Answer:","").replace("Correct Answer:","").replace("answer:","").replace("Question:","").replace("?","").replace("The correct answer is","").replace("1. ","").replace("Incorrect Answer","")
        generated.replace("-","").replace("What","").replace("The name of","").replace("Who","").replace("Question","").replace("#","")

        generated = generated.strip()
        if len(generated)>1 and "2" ==generated[-1] and not generated.isdigit():
            generated = generated[:-1]
        return generated