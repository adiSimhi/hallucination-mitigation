"""
This class is used to create the dataset from the DisentQA dataset.
It will create a labled hallucination and non hallucination dataset. given if the model prefer the
contextual answer or the parametric one
"""
import json
import sys

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import csv
import gzip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateDataset():
    def __init__(self, path_data_initial, threshold=1, model_name="GOAT-AI/GOAT-7B-Community", hall_save_path=None,
                 non_hall_save_path=None, general_save_path=None):
        set_seed(42)
        torch.manual_seed(42)
        MODEL_NAME = model_name
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.initial_dataset = self.create_initial_dataset(path_data_initial)
        self.threshold = threshold
        print(
            f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        print(
            f"2:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

        # self.model.to(device)
        print(
            f"3:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}",
            flush=True)
        self.model.eval()
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.eos_token
        self.dataset = []
        self.labels = []
        self.hall_save_path = hall_save_path
        self.non_hall_save_path = non_hall_save_path
        self.general_save_path = general_save_path
        print(
            f"hall_save_path={hall_save_path} non_hall_save_path={non_hall_save_path} general_save_path={general_save_path}")
        self.non_hall_dataset, self.hall_dataset, self.general_dataset = self.generate_final_dataset_using_model_using_ranks(
            self.model, self.initial_dataset)

    def create_initial_dataset(self, path):
        """
        use only  <p> type context and only counterfactual context,remove examples with number of tokens > 1000 or with
        more than 5 tokens in the target
        :param path:
        :return:
        """
        # load csv file
        csv.field_size_limit(sys.maxsize)
        data = []
        prompt_to_long = 0
        context_answer_to_long = 0
        parametric_answer_to_long = 0
        not_p = 0
        td = 0
        number_of_examples = 0
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for i, row in enumerate(spamreader):
                number_of_examples += 1
                if i == 0:
                    continue
                # print(row)
                if "</Td>" in row[-2] or "</Td>" in row[-2]:
                    td += 1
                    continue
                if "<P>" not in row[-2]:
                    not_p += 1
                    continue
                if row[7] == "factual":
                    continue
                prompt = "question: " + row[2] + "?\nanswer:"
                paraphrase_prompt = row[-2].replace("<P>", "").replace("</P>", "") + "\nanswer:"

                old_target = row[-1].split("\nparametric:")[1]
                new_target = row[-1].split("\nparametric:")[0]
                new_target = new_target.replace("contextual:", "")
                old_token = self.tok(old_target)["input_ids"][
                            1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "mistral" in self.model_name else \
                    self.tok(old_target)["input_ids"]
                new_token = self.tok(new_target)["input_ids"][
                            1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "mistral" in self.model_name else \
                    self.tok(new_target)["input_ids"]
                # print(f"{old_token=} {new_token=}")
                if self.tok([paraphrase_prompt], return_tensors="pt")["input_ids"].shape[1] > 1000 or len(
                        old_token) > 5 or len(new_token) > 5:
                    if self.tok([paraphrase_prompt], return_tensors="pt")["input_ids"].shape[1] > 1000:
                        prompt_to_long += 1
                    if len(old_token) > 5:
                        context_answer_to_long += 1
                    if len(new_token) > 5:
                        parametric_answer_to_long += 1
                    # print("too long")
                    continue
                # print([prompt, paraphrase_prompt, old_target, new_target, old_token, new_token])
                data.append([prompt, paraphrase_prompt, old_target, new_target, old_token, new_token])
        print(
            f"{prompt_to_long=} {context_answer_to_long=} {parametric_answer_to_long=} {len(data)=} {number_of_examples=} {not_p=} {td=}")
        return data

    def batch_generation_with_temperature(self, model, prompt, temperature=0.8):
        """
        generate 5 examples with the same prompt and return the generated texts
        :param model:
        :param prompt:
        :param temperature:
        :return:
        """
        input_ids = \
            self.tok([prompt for i in range(5)], padding=True, return_token_type_ids=False, return_tensors="pt")[
                "input_ids"].to(device)
        # run the same prompt 5 times in a batch
        with torch.no_grad():
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + 20), do_sample=True,
                                       pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature)
        generated = self.tok.batch_decode(model_out, skip_special_tokens=True)
        print(f"generated with temp={generated}")
        return generated

    def generate_final_dataset_using_model_using_ranks(self, model, initial_dataset):
        """
        generate the labeled dataset
        """
        nonhallucination_dataset = []
        hallucination_dataset = []
        general_dataset = []
        i = 0
        for point in initial_dataset:
            if i % 100 == 0:
                torch.cuda.empty_cache()
                print(
                    f"finished {i} points from the initial dataset with {len(nonhallucination_dataset)} and {len(hallucination_dataset)}\n\n")
                print(
                    f"10:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            if i % 1000 == 0 and i > 0:
                # save the dataset
                self.non_hall_dataset = nonhallucination_dataset
                self.hall_dataset = hallucination_dataset
                self.general_dataset = general_dataset
                self.save_data(self.non_hall_dataset, self.non_hall_save_path)
                self.save_data(self.hall_dataset, self.hall_save_path)
                self.save_data(self.general_dataset, self.general_save_path)
            i += 1
            prompt = point[0]
            praphraze_prompt = point[1]
            old_target = point[2]
            new_target = point[3]
            old_token = point[4]
            new_token = point[5]
            # print(f"{old_token=} {new_token=}")
            print(f"i={i}")
            print(f"{old_target=} {new_target=} {prompt=} {praphraze_prompt=}")
            answer_tokens = (old_token, new_token)
            print(
                f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

            rank_diff = self.get_rank_diff(model, praphraze_prompt, answer_tokens)
            print(
                f"2:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            print(f"running on the original prompt {prompt}")
            temp_generation = self.batch_generation_with_temperature(model, prompt, temperature=0.8)
            temp_generation = [temp_generation[i][len(prompt):] for i in range(len(temp_generation))]
            count_know = 0
            old_target_in_generated_text = False
            for temp in temp_generation:
                if old_target.strip().lower() in temp.lower() or old_target.lower() in temp.lower():
                    count_know += 1
            if count_know >= 4:
                old_target_in_generated_text = True
            print(f"{old_target_in_generated_text=} {temp_generation=}")
            print(
                f"3:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

            print(f"{rank_diff=}")
            # If the modified was able to change the model's prediction
            fact_info = (prompt, old_target, new_target, old_token, new_token, praphraze_prompt,
                         count_know, rank_diff,
                         )
            print(f"{fact_info=}")
            if rank_diff <= -self.threshold and old_target_in_generated_text:
                nonhallucination_dataset.append(fact_info)
            elif rank_diff >= self.threshold and old_target_in_generated_text:
                hallucination_dataset.append(
                    fact_info)
            else:
                general_dataset.append(fact_info)
        print(
            f"finished creating final dataset with {len(nonhallucination_dataset)} and {len(hallucination_dataset)} {len(general_dataset)} using {i} points from the initial dataset\n\n")
        return nonhallucination_dataset, hallucination_dataset, general_dataset

    def get_rank_diff(self, model, prompt, answer_tokens):
        """
        get the rank difference between the two answers
        """
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        # print(f"{input_ids=}")
        with torch.no_grad():
            model_out = model(input_ids)
        logits = model_out.logits
        probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        top_k = torch.topk(probabilities[0], k=5)
        top_k_tokens = top_k.indices
        top_k_tokens = self.tok.convert_ids_to_tokens(top_k_tokens)
        print(
            f"{top_k_tokens=} {top_k=} {answer_tokens=} {top_k.indices=} {top_k.values=} {self.tok.convert_ids_to_tokens(answer_tokens[0])=} {self.tok.convert_ids_to_tokens(answer_tokens[1])=}")
        del top_k, top_k_tokens
        # probability of the first token of the old and new answer
        prob_old = [probabilities[0][answer_tokens[0][0]].item()]
        prob_new = [probabilities[0][answer_tokens[1][0]].item()]
        rank_old = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[0]])]
        rank_new = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[0]])]
        print(f" {answer_tokens=}")
        # print(f"{prob_old=} {prob_new=}")
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        for i, token in enumerate(answer_tokens[0][:-1]):
            # add the token to the input_ids
            next_token = answer_tokens[0][i + 1]
            input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
            with torch.no_grad():
                model_out = model(input_ids)
            logits = model_out.logits
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            prob_old.append(probabilities[0][next_token].item())
            rank_old.append(sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[-1]]))
        # print(f"{prob_old=}")
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        for i, token in enumerate(answer_tokens[1][:-1]):
            next_token = answer_tokens[1][i + 1]
            input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
            with torch.no_grad():
                model_out = model(input_ids)
            logits = model_out.logits
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            prob_new.append(probabilities[0][next_token].item())
            rank_new.append(sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[-1]]))
        # the index of the first token that is not the same in the old and new answer
        shorter_answer = min(len(answer_tokens[0]), len(answer_tokens[1]))
        non_similar_index = [i for i in range(shorter_answer) if answer_tokens[0][i] != answer_tokens[1][i]]
        if len(non_similar_index) == 0:
            non_similar_index = [0]
        rank = rank_new[non_similar_index[0]] - rank_old[non_similar_index[0]]
        print(f"{prob_old=}, {prob_new=} {rank_old=} {rank_new=} {non_similar_index=} {rank=}")
        if abs(rank) > self.threshold:
            print(f"the two answers are different {rank=}")
        del input_ids
        del model_out
        torch.cuda.empty_cache()
        if answer_tokens[0] == answer_tokens[1]:
            assert prob_old == prob_new
            print(f"the two answers are the same {prob_old=}")
            return prob_old
        print(f"answer_tokens[0]={answer_tokens[0]} answer_tokens[1]={answer_tokens[1]}")
        return rank





    def save_data(self,data,path):
        with open(path, "w") as f:
            json.dump(data, f)