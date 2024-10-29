"""
This file is used to create the dataset for the hallucination detection task without context.
"""
import json
import random
import sys

import datasets
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
        self.initial_dataset = self.create_initial_dataset_for_trivia_qa(path_data_initial)
        self.threshold = threshold
        print(f"{self.threshold}=")
        print(
            f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        print(
            f"2:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")

        # self.model.to(device)
        print(
            f"3:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
        self.model.eval()
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.eos_token
        self.dataset = []
        self.labels = []
        self.hall_save_path = hall_save_path
        self.non_hall_save_path = non_hall_save_path
        self.general_save_path = general_save_path
        self.non_hall_dataset, self.hall_dataset, self.general_dataset = self.generate_final_dataset_using_model_using_ranks(
            self.model, self.initial_dataset)



    def create_initial_dataset_for_trivia_qa(self, path):
        """
        create the initial dataset for the hallucination detection task for triviaqa
        :param path:
        :return:
        """
        print("creating initial dataset for trivia qa")
        # dataset
        dataset = datasets.load_dataset("trivia_qa", 'rc', ignore_verifications=True)
        train, validation, test = dataset["train"], dataset["validation"], dataset["test"]
        dataset = train
        print(f"the length of the dataset is {len(dataset)}")
        random.seed(42)
        index = random.sample(range(len(dataset)), 20000)
        dataset = dataset.select(index)
        assert len(dataset) == 20000
        data = []
        for i, row in enumerate(dataset):
            prompt = "question: " + row["question"] + "\nanswer:"
            old_target = row["answer"]["value"]
            old_target = old_target
            old_token = self.tok(old_target)["input_ids"][
                        1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name \
                               or "mistral" in self.model_name else \
                self.tok(old_target)["input_ids"]
            if len(
                    old_token) > 5:
                # print("too long")
                continue
            if i < 10:
                print([prompt, old_target, old_token])
            data.append([prompt, old_target, old_token])
        print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def generate_final_dataset_using_model_using_ranks(self, model, initial_dataset):
        """
        generate final dataset using model final probabilities
        create two datasets one with logits diff of old answer - new answer > threshold and one with logits diff of
        old answer - new answer < -threshold
        examples in this dataset have the following format:
        prompt, old_target, new_target, old_token, new_token, praphraze_prompt, final_logits,logits_on_true_answer_without_context

        :param model:
        :param initial_dataset:
        :return:both datasets (nonhallucination_dataset,hallucination_dataset)
        """
        nonhallucination_dataset = []
        hallucination_dataset = []
        general_group = []
        pp_hallucination_dataset = []
        pp_nonhallucination_dataset = []
        i = 0
        self.list_good_shot = [
            "question: What is the capital of France?\nanswer: Paris\n",
            "question: How many continents are there?\nanswer: 7\n",
            "question: Who wrote 'Romeo and Juliet'?\nanswer: William Shakespeare\n",
            "question: What is the square root of 64?\nanswer: 8\n",
            "question: Which element has the chemical symbol 'H'?\nanswer: Hydrogen\n",
            "question: Who was the first President of the United States?\nanswer: George Washington\n",
            "question: What is the powerhouse of the cell?\nanswer: Mitochondria\n",
            "question: In what year did World War II end?\nanswer: 1945\n",
            "question: What is the currency of Japan?\nanswer: Japanese Yen\n",
            "question: Who painted the Mona Lisa?\nanswer: Leonardo da Vinci\n",
            "question: What is the speed of light?\nanswer: 299,792 kilometers per second\n",
            "question: How many sides does a hexagon have?\nanswer: 6\n",
            "question: What is the boiling point of water in Celsius?\nanswer: 100 degrees\n",
            "question: Who wrote 'To Kill a Mockingbird'?\nanswer: Harper Lee\n",
            "question: What is the capital of Australia?\nanswer: Canberra\n",
            "question: What is the largest ocean on Earth?\nanswer: Pacific Ocean\n",
            "question: Who discovered penicillin?\nanswer: Alexander Fleming\n",
            "question: What is the chemical symbol for gold?\nanswer: Au\n",
            "question: What is the smallest prime number?\nanswer: 2\n",
            "question: How many planets are there in our solar system?\nanswer: 8\n"]
        self.list_bad_shot = [
            "question: What is the capital of France?\nanswer: Berlin\n",
            "question: How many continents are there?\nanswer: 6\n",
            "question: Who wrote 'Romeo and Juliet'?\nanswer: Jane Austen\n",
            "question: What is the square root of 64?\nanswer: 7\n",
            "question: Which element has the chemical symbol 'H'?\nanswer: Helium\n",
            "question: Who was the first President of the United States?\nanswer: Abraham Lincoln\n",
            "question: What is the powerhouse of the cell?\nanswer: Golgi Apparatus\n",
            "question: In what year did World War II end?\nanswer: 1939\n",
            "question: What is the currency of Japan?\nanswer: Euro\n",
            "question: Who painted the Mona Lisa?\nanswer: Pablo Picasso\n",
            "question: What is the speed of light?\nanswer: 300,000 kilometers per second\n",
            "question: How many sides does a hexagon have?\nanswer: 5\n",
            "question: What is the boiling point of water in Celsius?\nanswer: 50 degrees\n",
            "question: Who wrote 'To Kill a Mockingbird'?\nanswer: J.K. Rowling\n",
            "question: What is the capital of Australia?\nanswer: Sydney\n",
            "question: What is the largest ocean on Earth?\nanswer: Atlantic Ocean\n",
            "question: Who discovered penicillin?\nanswer: Isaac Newton\n",
            "question: What is the chemical symbol for gold?\nanswer: Ag\n",
            "question: What is the smallest prime number?\nanswer: 1\n",
            "question: How many planets are there in our solar system?\nanswer: 9\n",
        ]
        random.seed(42)
        average_rank = []
        count_hall_prefer_parametric = 0
        count_nonhall_prefer_parametric = 0

        for point in initial_dataset:
            if i % 100 == 0:
                torch.cuda.empty_cache()
                print(
                    f"finished {i} points from the initial dataset with {len(nonhallucination_dataset)} and {len(hallucination_dataset)}\n\n")
                print(
                    f"10:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            if i % 1000 == 0:
                # save the dataset
                self.non_hall_dataset = nonhallucination_dataset
                self.hall_dataset = hallucination_dataset
                self.general_dataset = general_group
                self.save_data(self.non_hall_dataset, self.non_hall_save_path)
                self.save_data(self.hall_dataset, self.hall_save_path)
                self.save_data(self.general_dataset, self.general_save_path)
                print(f"average rank={np.average(average_rank)}+- {np.std(average_rank)}")
                print(
                    f"pp_hallucination_dataset={np.average(pp_hallucination_dataset)}+- {np.std(pp_hallucination_dataset)} pp_nonhallucination_dataset={np.average(pp_nonhallucination_dataset)}+- {np.std(pp_nonhallucination_dataset)}")
                print(
                    f"count_hall_prefer_parametric={count_hall_prefer_parametric} count_nonhall_prefer_parametric={count_nonhall_prefer_parametric}")
            i += 1
            prompt = point[0]
            prompt = prompt
            old_target = point[1]
            old_token = point[2]
            # print(f"{old_token=} {new_token=}")
            print(f"i={i}")
            answer_tokens = (old_token, old_token)
            print(
                f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            print(
                f"2:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            print(f"running on the original prompt {prompt} and the old target {old_target}")
            index_of_shots = random.sample(range(len(self.list_bad_shot)), 3)
            bad_shot = self.list_bad_shot[index_of_shots[0]] + self.list_bad_shot[index_of_shots[1]] + \
                       self.list_bad_shot[index_of_shots[2]]
            good_shot = self.list_good_shot[index_of_shots[0]] + self.list_good_shot[index_of_shots[1]] + \
                        self.list_good_shot[index_of_shots[2]]
            print(f"{bad_shot=}")
            rank = self.get_rank(model,
                                 (bad_shot + prompt).replace("\nanswer:", "\nwrong answer:"),
                                 answer_tokens)
            rank_reg = self.get_rank(model, (good_shot + prompt), answer_tokens)
            average_rank.append(rank)
            print(f"{prompt=} {rank=} {rank>self.threshold=} {old_target=}")
            temp_generation = self.batch_generation_with_temperature(model, prompt, temperature=0.8)
            temp_generation = [temp_generation[i][len(prompt):] for i in range(len(temp_generation))]
            assert len(temp_generation) == 5
            know_answer = False
            count_know = 0
            for temp in temp_generation:
                if old_target.strip().lower() in temp.lower() or old_target.lower() in temp.lower():
                    count_know += 1
            if count_know >= 4:
                know_answer = True
            print(f"{know_answer=} {count_know=} {rank_reg=} {rank=}")

            # If the modified was able to change the model's prediction
            fact_info = (prompt, old_target, old_target, old_token, old_token,
                         (bad_shot + prompt).replace("\nanswer:", "\nwrong answer:"), count_know,
                         rank - rank_reg
                         )
            print(f"{fact_info=}")
            # it is not hallucinate if the most prefer token is the same as the old token else it is hallucinate
            if know_answer and rank - rank_reg == 0:
                nonhallucination_dataset.append(fact_info)
                pp_nonhallucination_dataset.append(rank)
            elif know_answer and rank - rank_reg >= self.threshold and rank_reg == 0:
                hallucination_dataset.append(
                    fact_info)
                pp_hallucination_dataset.append(rank)
            else:
                general_group.append(fact_info)
        print(
            f"finished creating final dataset with {len(nonhallucination_dataset)=} and {len(hallucination_dataset)=} and {len(general_group)=} using {i} points from the initial dataset\n\n")
        print(
            f"pp_hallucination_dataset={np.average(pp_hallucination_dataset)}+- {np.std(pp_hallucination_dataset)} pp_nonhallucination_dataset={np.average(pp_nonhallucination_dataset)}+- {np.std(pp_nonhallucination_dataset)}")
        print(f"average rank={np.average(average_rank)}+- {np.std(average_rank)}")
        print(
            f"count_hall_prefer_parametric={count_hall_prefer_parametric} count_nonhall_prefer_parametric={count_nonhall_prefer_parametric}")
        return nonhallucination_dataset, hallucination_dataset, general_group

    def get_rank(self, model, prompt, answer_tokens):
        """
        get the diff of the geometric mean of the probability of the answers given the prompt
        :param model:
        :param prompt:
        :param answer_tokens:
        :return: if less than zero prefer the new answer and if more than zero prefer the old answer
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
        rank_old = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[0]])]
        prob_old_index = sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[0]])
        print(f"{prob_old_index=} {answer_tokens=}")
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
        assert len(prob_old) == len(answer_tokens[0]) == len(rank_old), f"{len(prob_old)=} {len(answer_tokens[0])=} "
        prob_old_score = np.prod(np.array(prob_old)) ** (1 / len(prob_old))
        rank_old_score = sum(rank_old) / len(rank_old)
        print(f"{prob_old_score=} {rank_old_score=}")
        del input_ids
        del model_out
        torch.cuda.empty_cache()
        return rank_old_score


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


    def save_data(self,data,path):
        with open(path, "w") as f:
            json.dump(data, f)
