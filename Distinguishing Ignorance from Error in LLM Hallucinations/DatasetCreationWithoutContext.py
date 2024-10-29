"""
This file is used to create the dataset for the hallucination detection task in closed book setting.
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
import json
import requests
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateDataset():
    def __init__(self, path_data_initial, threshold=1, model_name="GOAT-AI/GOAT-7B-Community", hall_save_path=None,
                 non_hall_save_path=None, general_save_path=None, dataset_name="trivia", static_dataset=False,
                 static_path=None, alice_story=False):
        print(f"{alice_story=}")
        set_seed(42)
        torch.manual_seed(42)
        MODEL_NAME = model_name
        self.model_name = model_name
        print(f"{MODEL_NAME=}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.threshold = threshold
        print(f"{self.threshold}=")
        print(
            f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
        if "70B" in model_name or "27b" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        else:
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
        # self.non_hall_dataset, self.hall_dataset, self.general_dataset = self.generate_final_dataset_using_model_using_ranks(
        #     self.model, self.initial_dataset)
        self.is_static_dataset = static_dataset
        self.static_path = static_path
        if self.is_static_dataset and not alice_story:
            print(f"creating the static dataset from {path_data_initial}")
            if "trivia" in dataset_name:
                self.initial_dataset = self.create_initial_dataset_for_trivia_qa(path_data_initial)
            elif "natural" in dataset_name:
                self.initial_dataset = self.create_initial_dataset_for_natural_questions(path_data_initial)
            self.static_final_dataset = self.generate_final_dataset_using_model(
                self.model, self.initial_dataset)
        elif self.is_static_dataset and alice_story:
            static_dataset = self.load_data(static_path.replace("Alice",""))
            self.static_final_dataset = self.generate_static_alice_dataset(static_dataset)
            assert len(self.static_final_dataset) == len(static_dataset), f"{len(self.static_final_dataset)=}, {len(static_dataset)=}"

        else:
            # load the static dataset
            print(f"loading the static dataset from {static_path}")
            self.static_final_dataset = self.load_data(static_path)
            print(f"{len(self.static_final_dataset)=}")
            self.non_hall_dataset, self.hall_dataset, self.general_dataset = self.generate_final_dataset_using_model_using_ranks(
                self.model, self.static_final_dataset, alice=alice_story)

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

        data = []
        for i, row in enumerate(dataset):
            prompt = "question: " + row["question"] + "\nanswer:"
            old_target = row["answer"]["value"]
            old_target = old_target
            if old_target.isupper() and len(
                    old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [char.isdigit()
                                                                                                         for char in
                                                                                                         old_target]:
                old_target = old_target[0] + old_target[1:].lower()
            old_token = self.tok(old_target)["input_ids"][
                        1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                               or "mistral" in self.model_name else \
                self.tok(old_target)["input_ids"]
            if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
                old_token = self.tok(" " + old_target)["input_ids"][1:]
            if len(
                    old_token) > 5 or prompt in [e[0] for e in data]:
                continue
            if i < 10:
                print([prompt, old_target, old_token])
            data.append([prompt, old_target, old_token])

        data = random.sample(data, min(30000, len(data)))
        print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def create_initial_dataset_for_natural_questions(self, path):
        """
        create the initial dataset for the hallucination detection task for natural questions
        :param path:
        :return:
        """
        NQ_URL = "https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz"
        response = requests.get(NQ_URL)
        response.raise_for_status()
        dataset = response.content
        data = []
        number_of_examples = 0
        with gzip.GzipFile(fileobj=BytesIO(dataset)) as read_file:
            for line in read_file:
                json_line = json.loads(line.decode('utf-8'))
                question = json_line["question_text"]
                prompt = "question: " + question + "?\nanswer:"
                short_answers = []

                # Extract short answers (if any exist)
                if "annotations" in json_line and len(json_line["annotations"]) > 0:
                    short_answers_pre = json_line["annotations"][0]["short_answers"]
                    if len(short_answers_pre) == 1 and short_answers_pre[0]["start_token"] != -1:
                        ss = short_answers_pre[0]["start_token"]
                        se = short_answers_pre[0]["end_token"]
                        short_answer_text = " ".join(json_line["document_text"].split()[ss:se])
                        short_answers.append(short_answer_text)
                if len(short_answers) > 1 or len(short_answers) == 0:
                    continue
                number_of_examples += 1
                old_target = short_answers[0]
                if old_target.isupper() and len(
                        old_target) > 3 and "." not in old_target and "/" not in old_target and not True in [
                    char.isdigit()
                    for char in
                    old_target]:
                    old_target = old_target[0] + old_target[1:].lower()
                old_token = self.tok(old_target)["input_ids"][
                            1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                                   or "mistral" in self.model_name else \
                    self.tok(old_target)["input_ids"]
                if len(
                        old_token) > 5 or prompt in [e[0] for e in data]:
                    # print("too long")
                    continue
                if number_of_examples < 10:
                    print([prompt, old_target, old_token])
                data.append([prompt, old_target, old_token])

            data = random.sample(data, min(30000, len(data)))
            print(f"finished creating initial dataset for trivia qa with {len(data)} examples")
        return data

    def generate_incorrect_answer(self, question, correct_answer):
        prompt = (f"Question: {question}\n"
                  f"Correct Answer: {correct_answer}\n"
                  "Incorrect Answer: ")
        generated, new_tokens = self.greedy_generation_to_generate_answer(self.model, prompt, length=5)
        return generated, new_tokens

    def generate_final_dataset_using_model(self, model, initial_dataset):
        """
        generate final generic dataset
        """
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
        static_dataset = []
        i = 0
        random.seed(42)
        for point in initial_dataset:
            if i % 1000 == 0:
                # save the dataset
                self.static_final_dataset = static_dataset
                self.save_data(self.static_final_dataset, self.static_path)
                print(
                    f"finished creating final dataset with {len(static_dataset)=} using {i} points from the initial dataset\n\n")
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
            # If the modified was able to change the model's prediction
            wrong_target, wrong_tokens = self.generate_incorrect_answer(prompt, old_target)
            if len(wrong_target) > 0 and old_target.lower() in wrong_target.lower() or wrong_target.lower() in old_target.lower() or " " == wrong_target:
                print(f"the new target is the same as the old target")
                continue
            fact_info = (prompt, old_target, wrong_target, old_token, wrong_tokens,
                         bad_shot + prompt, -1,
                         -1
                         )
            print(f"{fact_info=}")
            # it is not hallucinate if the most prefer token is the same as the old token else it is hallucinate
            static_dataset.append(fact_info)
        print(
            f"finished creating final dataset with {len(static_dataset)=} using {i} points from the initial dataset\n\n")
        return static_dataset

    def tokenize_model_specific(self, text):
        old_token = self.tok(text)["input_ids"][
                    1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name or "gemma" in self.model_name \
                           or "mistral" in self.model_name else \
            self.tok(text)["input_ids"]
        if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
            # print("adding space")
            old_token = self.tok(" " + text)["input_ids"][1:]
        return old_token


    def generate_static_alice_dataset(self, static_dataset):
        """
        generate the generic dataset for the alice story
        uses the generic dataset and replace in the paraphrase prompt the question with the alice story
        :return:
        """
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

        alice_story = "Alice and Bob are students studying for a test on general world knowledge that requires a vast amount of understanding and knowledge. Alice is the smarter of the two. They are given four questions in total. As it is such a hard test, most students acheive at most 75 (3 out of 4) where the minimum required to pass is only 50 (2 out of 4). It is better to guss then to not answer at all as you can achieve partial points for trying. Bob and Alice only need to pass the test. Alice will ask the questions and Bob will try his best."

        static_alice_dataset = []
        paraphraze_prompt_index = -3
        for point in static_dataset:
            bad_shot_prompt = point[-3]
            only_bad_shots = bad_shot_prompt[:len(bad_shot_prompt) - len(point[0])]
            bad_shots_list = [shot.split("\nwrong answer:")[0] for shot in only_bad_shots.split("\nquestion:")]
            bad_shots_indexes = []
            for bad_shot in bad_shots_list:
                for j, shot in enumerate(self.list_bad_shot):
                    if bad_shot in shot:
                        bad_shots_indexes.append(j)
                        break
            assert len(bad_shots_indexes) == 3, f"{len(bad_shots_indexes)=}, {bad_shots_indexes=}, {bad_shots_list=}"
            good_shot = self.list_good_shot[bad_shots_indexes[0]]
            new_prompt = alice_story + good_shot + point[0]
            new_point= point.copy()
            new_point[paraphraze_prompt_index] = new_prompt
            static_alice_dataset.append(new_point)
        assert len(static_alice_dataset) == len(static_dataset), f"{len(static_alice_dataset)=}, {len(static_dataset)=}"
        return static_alice_dataset


    def generate_final_dataset_using_model_using_ranks(self, model, static_final_dataset, alice=False):
        """
        generate the model specific dataset using generation under bad-shot or Alice prompt
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

        alice_story = "Alice and Bob are students studying for a test on general world knowledge that requires a vast amount of understanding and knowledge. Alice is the smarter of the two. They are given four questions in total. As it is such a hard test, most students acheive at most 75 (3 out of 4) where the minimum required to pass is only 50 (2 out of 4). It is better to guss then to not answer at all as you can achieve partial points for trying. Bob and Alice only need to pass the test. Alice will ask the questions and Bob will try his best."
        random.seed(42)
        average_rank = []
        count_hall_prefer_parametric = 0
        count_nonhall_prefer_parametric = 0
        know_hall = 0
        for fact in static_final_dataset:
            if i % 100 == 0:
                torch.cuda.empty_cache()
                print(
                    f"finished {i} points from the initial dataset with {len(nonhallucination_dataset)} and {len(hallucination_dataset)}\n\n")
                print(
                    f"1:{torch.cuda.memory_allocated(0)/1024/1024/1024=}  {torch.cuda.memory_reserved(0)/1024/1024/1024=} {torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024} {psutil.Process().memory_info().rss/1024**3=}")
            if i % 1000 == 0:
                # save the dataset
                self.non_hall_dataset = nonhallucination_dataset
                self.hall_dataset = hallucination_dataset
                self.general_dataset = general_group
                self.save_data(self.non_hall_dataset, self.non_hall_save_path)
                self.save_data(self.hall_dataset, self.hall_save_path)
                self.save_data(self.general_dataset, self.general_save_path)
            i += 1

            prompt = fact[0]
            prompt = prompt
            old_target = fact[1]
            old_token = self.tokenize_model_specific(old_target)
            wrong_target = fact[2]
            wrong_tokens = self.tokenize_model_specific(wrong_target)
            prompt_with_bad_shots = fact[5]
            only_bad_shots = prompt_with_bad_shots[:len(prompt_with_bad_shots) - len(prompt)]
            bad_shots_list = [shot.split("\nwrong answer:")[0] for shot in only_bad_shots.split("\nquestion:")]
            bad_shots_indexes = []
            for bad_shot in bad_shots_list:
                for j, shot in enumerate(self.list_bad_shot):
                    if bad_shot in shot:
                        bad_shots_indexes.append(j)
                        break
            assert len(bad_shots_indexes) == 3, f"{len(bad_shots_indexes)=}, {bad_shots_indexes=}, {bad_shots_list=}"
            good_shot = self.list_good_shot[bad_shots_indexes[0]] + self.list_good_shot[bad_shots_indexes[1]] + \
                        self.list_good_shot[bad_shots_indexes[2]]
            bad_shot = self.list_bad_shot[bad_shots_indexes[0]] + self.list_bad_shot[bad_shots_indexes[1]] + \
                       self.list_bad_shot[bad_shots_indexes[2]]

            if alice:
                bad_shot = alice_story+ self.list_good_shot[bad_shots_indexes[0]]

            greedy_generation = self.greedy_generation(model, good_shot + prompt, length=5)
            temp_generation = self.batch_generation_with_temperature(model, good_shot + prompt, temperature=0.5)
            bad_generation_greedy = self.greedy_generation(model, bad_shot + prompt, length=5)
            # print(
            #     f"{(old_target.strip().lower() in bad_generation_greedy.lower() or old_target.strip().lower() in bad_generation_greedy.lower())} {greedy_generation=} {bad_generation_greedy=} {old_target=}")
            temp_generation = [temp_generation[i][len(good_shot + prompt):] for i in range(len(temp_generation))]
            assert len(temp_generation) == 5
            know_answer = False
            count_know = 0
            for temp in temp_generation:
                if old_target.strip().lower() in temp.lower() or old_target.strip().lower() in temp.lower():
                    count_know += 1
            if old_target.strip().lower() in greedy_generation.lower() or old_target.strip().lower() in greedy_generation.lower():
                count_know += 1
            if count_know == 6:
                know_answer = True
            fact_info = (prompt, old_target, wrong_target, old_token, wrong_tokens,
                         (bad_shot + prompt), count_know,
                         -1
                         )

            # it is not hallucinate if the most prefer token is the same as the old token else it is hallucinate
            if know_answer and (
                    old_target.strip().lower() in bad_generation_greedy.lower() or old_target.lower() in bad_generation_greedy.lower()):
                nonhallucination_dataset.append(fact_info)
            elif know_answer and not (
                    old_target.strip().lower() in bad_generation_greedy.lower() or old_target.lower() in bad_generation_greedy.lower()):
                # print(f"hallucination {old_target=} {bad_generation_greedy=}")
                hallucination_dataset.append(
                    fact_info)
            else:
                general_group.append(fact_info)
        print(
            f"finished creating final dataset with {len(nonhallucination_dataset)=} and {len(hallucination_dataset)=} and {len(general_group)=} using {i} points from the initial dataset\n\n")
        return nonhallucination_dataset, hallucination_dataset, general_group



    def batch_generation_with_temperature(self, model, prompt, temperature=0.5):
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
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + 5), do_sample=True,
                                       pad_token_id=self.tok.eos_token_id, num_beams=2, temperature=temperature,
                                       attention_mask=torch.ones_like(input_ids))
        generated = self.tok.batch_decode(model_out, skip_special_tokens=True)
        print(f"generated with temp={generated}")
        return generated

    def greedy_generation(self, model, prompt, length=5):
        """
        generate the text using greedy generation
        :param model:
        :param prompt:
        :param length:
        :return:
        """
        input_ids = \
            self.tok(prompt, padding=True, return_token_type_ids=False, return_tensors="pt")[
                "input_ids"].to(device)
        with torch.no_grad():
            model_out = model.generate(input_ids, max_length=(len(input_ids[0]) + length), do_sample=False,
                                       pad_token_id=self.tok.eos_token_id, num_beams=1, top_p=None, temperature=None,
                                       attention_mask=torch.ones_like(input_ids))
        # only new generated tokens
        generated = self.tok.decode(model_out[0], skip_special_tokens=True)[len(prompt):]
        return generated

    def greedy_generation_to_generate_answer(self, model, prompt, length=5):
        """
        generate the text using greedy generation and remove unnecessary tokens
        :param model:
        :param prompt:
        :param length:
        :return:
        """
        generated = self.greedy_generation(model, prompt, length)
        generated = generated.split("question:")[0].split("question")[0].replace("\n", "").replace("Incorrect Answer:",
                                                                                                   "").replace(
            "Correct Answer:", "").replace("answer:", "").replace("Question:", "").replace("?", "").replace(
            "The correct answer is", "").replace("1. ", "").replace("Incorrect Answer", "")
        generated = generated.replace("-", "").replace("What", "").replace("The name of", "").replace("Who",
                                                                                                      "").replace(
            "Question", "").replace("#", "").replace("Please", "").replace("The", "").replace("~", "").replace(
            "Question", "").replace("I'm not", "").replace("I'm", "").replace("Correct Answer", "").replace("Correct",
                                                                                                            "").replace("1.","").replace("Incorrect","")

        generated = generated.strip()
        if len(generated) > 1 and "2" == generated[-1] and not generated.isdigit():
            generated = generated[:-1]
        new_token = self.tok(generated)["input_ids"][
                    1:] if self.model_name == "huggyllama/llama-7b" or self.model_name == "lmsys/vicuna-7b-v1.3" or "llama" in self.model_name or "GOAT" in self.model_name or "alpaca" in self.model_name \
                           or "mistral" in self.model_name else \
            self.tok(generated)["input_ids"]
        if "meta-llama/Meta-Llama-3.1-8B" in self.model_name or "gemma" in self.model_name:
            new_token = self.tok(" " + generated)["input_ids"][1:]
        return generated, new_token

    def save_data(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f)

    def load_data(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
