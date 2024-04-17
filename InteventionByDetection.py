"""
This file is responsible for the intervention by detection
"""

import functools
import gc
import json
import os
import pickle
import subprocess
import time

import psutil
from torch.nn import CrossEntropyLoss
from typing import List, Tuple

import numpy as np
import torch
import transformers

from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from models_config import get_model_config
import ModelInside

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random

set_seed(42)
torch.manual_seed(42)
generation_length = 20


class InterventionByDetection():

    def __init__(self, path_to_results, data_path_without_hallucinations, data_path_with_hallucinations, model_name,
                 dataset_size, dataset_name, threshold_of_data, use_mlp, use_attention, use_heads, use_residual,
                 alpha=5, static_intervention=False, concatenate_answer=False,
                 on_test_set=False, seed_train_val=None, general_data_path=None,
                 use_classifier_for_intervention=False):
        print(f"{transformers.__version__=}")
        self.seed_train_val = seed_train_val
        print(f"{self.seed_train_val=}")
        self.model_name = model_name
        self.intervention = None
        self.alpha = alpha
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.hallucinations_examples = self.load_dataset(data_path_with_hallucinations)
        self.non_hallucinations_examples = self.load_dataset(data_path_without_hallucinations)
        self.general_examples = self.load_dataset(general_data_path) if general_data_path is not None else None
        self.path_to_save_results = path_to_results + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold_of_data}/concat_answer{False}_size{dataset_size}/"
        start_time = time.time()
        MLPCheck = ModelInside.ModelInside(path_to_results,
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=threshold_of_data,concat_answer=False)
        self.all_mlp_vector_with_hall, self.all_attention_vector_with_all, self.all_mlp_vector_without_hall, self.all_attention_vector_without_hall, self.heads_vectors_with, self.heads_vectors_without, self.all_residual_with, self.all_residual_without = MLPCheck.load_all_data()
        print(f"load data time {time.time() - start_time}")
        print(f"{type(self.all_attention_vector_without_hall)=}")
        print(
            f"{len(self.all_mlp_vector_with_hall)=} {len(self.all_attention_vector_with_all)=} {len(self.all_mlp_vector_without_hall)=} {len(self.all_attention_vector_without_hall)=} {len(self.heads_vectors_with)=} {len(self.heads_vectors_without)=} {len(self.all_residual_with)=} {len(self.all_residual_without)=}")
        assert len(self.all_mlp_vector_with_hall) == len(self.all_attention_vector_with_all) == len(
            self.all_mlp_vector_without_hall) == len(self.all_attention_vector_without_hall) == len(
            self.heads_vectors_with) == len(self.heads_vectors_without) == len(self.all_residual_with) == len(
            self.all_residual_without)

        # create a classifier for each type of information - mlp,attention,residual,heads
        self.classifiers_dict, self.acc_classification_dict = self.get_classifier_dicts(use_mlp=use_mlp,
                                                                                        use_heads=use_heads,
                                                                                        use_attention=use_attention,
                                                                                        use_residual=use_residual)
        # create an intervention dict
        self.intervention_dict = self.get_intervention_dict(use_mlp=use_mlp, use_heads=use_heads,
                                                            use_attention=use_attention, use_residual=use_residual)
        start_time = time.time()
        self.intervention = ModelIntervention(path_to_results=path_to_results, model_name=model_name,
                                              dataset_size=dataset_size, dataset_name=dataset_name,
                                              threshold_of_data=threshold_of_data, use_mlp=use_mlp,
                                              use_attention=use_attention, use_heads=use_heads,
                                              use_residual=use_residual, intervention_all_dict=self.intervention_dict,
                                              classifier_dict=self.classifiers_dict,
                                              alpha=alpha,
                                              acc_classification_val=self.acc_classification_dict,
                                              static_intervention=static_intervention)
        print(f"intervention creation class time {time.time() - start_time}")
        self.generate_examples_for_intervention(on_test_set=on_test_set)

        # create an intervention dict using classifier
        if use_classifier_for_intervention:
            print(f"create an intervention dict using classifier")
            self.intervention_with_classifier = self.get_intervention_for_all_data_types_based_classifier(
                self.all_mlp_vector_with_hall,
                self.all_mlp_vector_without_hall,
                self.all_attention_vector_with_all,
                self.all_attention_vector_without_hall,
                self.all_residual_with,
                self.all_residual_without,
                self.heads_vectors_with,
                self.heads_vectors_without)
            self.intervention.all_interventions = self.intervention_with_classifier
            self.check_intervention_difference_between_intervention_dicts(self.intervention_with_classifier,
                                                                          self.intervention_dict)
        # create an intervention dict using concatenate answer examples (post answer)
        if concatenate_answer:
            self.get_intervention_concat(MLPCheck, use_mlp, use_attention, use_heads, use_residual)

    def get_classifier_dicts(self, use_mlp, use_attention, use_heads, use_residual):
        """
        get the classifier dict and the accuracy classification dict on the validation set
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        start_time = time.time()
        classifier_dict_path = self.path_to_save_results + f"classifier_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        acc_classification_dict_path = self.path_to_save_results + f"acc_classification_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        if self.seed_train_val is not None:
            classifier_dict_path = classifier_dict_path.replace(".pkl", f"seed_{self.seed_train_val}.pkl")
            acc_classification_dict_path = acc_classification_dict_path.replace(".pkl",
                                                                                f"seed_{self.seed_train_val}.pkl")
        print(f"{classifier_dict_path=} {acc_classification_dict_path=}")
        if os.path.exists(classifier_dict_path) and os.path.exists(acc_classification_dict_path):
            print(f"load classifier dict from {classifier_dict_path} and {acc_classification_dict_path}")
            # load the classifier dict from pickle
            with open(classifier_dict_path, 'rb') as handle:
                self.classifiers_dict = pickle.load(handle)
            with open(acc_classification_dict_path, 'rb') as handle:
                self.acc_classification_dict = pickle.load(handle)
        else:
            self.classifiers_dict, self.acc_classification_dict = self.create_classifier_for_each_type_of_info()
            # save the classifier dict to pickle
            with open(classifier_dict_path, 'wb') as handle:
                pickle.dump(self.classifiers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(acc_classification_dict_path, 'wb') as handle:
                pickle.dump(self.acc_classification_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{self.acc_classification_dict=}")
        print(f"create classifier dict time {time.time() - start_time}")
        return self.classifiers_dict, self.acc_classification_dict

    def get_intervention_dict(self, use_mlp, use_attention, use_heads, use_residual):
        """
        get the intervention dict, the direction to add to mitigate the hallucinations
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        start_time = time.time()
        intervention_dict_path = self.path_to_save_results + f"intervention_dict_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        if self.seed_train_val is not None:
            intervention_dict_path = intervention_dict_path.replace(".pkl", f"seed_{self.seed_train_val}.pkl")
        print(f"{intervention_dict_path=}")
        if os.path.exists(intervention_dict_path):
            print(f"load intervention dict from {intervention_dict_path}")
            # load the classifier dict from pickle
            with open(intervention_dict_path, 'rb') as handle:
                self.intervention_dict = pickle.load(handle)
        else:
            self.intervention_dict = self.get_intervention_for_all_data_types(self.all_mlp_vector_with_hall,
                                                                              self.all_mlp_vector_without_hall,
                                                                              self.all_attention_vector_with_all,
                                                                              self.all_attention_vector_without_hall,
                                                                              self.all_residual_with,
                                                                              self.all_residual_without,
                                                                              self.heads_vectors_with,
                                                                              self.heads_vectors_without)
            with open(intervention_dict_path, 'wb') as handle:
                pickle.dump(self.intervention_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"intervention dict time {time.time() - start_time}")
        return self.intervention_dict

    def get_intervention_concat(self, MLPCheck, use_mlp, use_attention, use_heads, use_residual):
        """
        get the intervention dict using concatenate answer examples (post-answer)
        :param MLPCheck:
        :param use_mlp:
        :param use_attention:
        :param use_heads:
        :param use_residual:
        :return:
        """
        print(f"create an intervention dict using concatenate answer examples")
        path_intervention_dict_concatenate_answer = self.path_to_save_results + f"intervention_dict_concat_{use_mlp}_{use_attention}_{use_heads}_{use_residual}.pkl"
        if self.seed_train_val is not None:
            path_intervention_dict_concatenate_answer = path_intervention_dict_concatenate_answer.replace(".pkl",
                                                                                                          f"seed_{self.seed_train_val}.pkl")
        if os.path.exists(path_intervention_dict_concatenate_answer):
            print(f"load intervention dict concat from {path_intervention_dict_concatenate_answer}")
            # load the classifier dict from pickle
            with open(path_intervention_dict_concatenate_answer, 'rb') as handle:
                self.intervention_dict_concatenate_answer = pickle.load(handle)

        else:
            MLPCheck.path_to_save_results = MLPCheck.path_to_save_results.replace("concat_answerFalse",
                                                                                  "concat_answerTrue")
            print(f"{MLPCheck.path_to_save_results=}")
            assert "concat_answerTrue" in MLPCheck.path_to_save_results
            self.all_mlp_vector_with_hall_concatenate_answer, self.all_attention_vector_with_all_concatenate_answer, self.all_mlp_vector_without_hall_concatenate_answer, self.all_attention_vector_without_hall_concatenate_answer, self.heads_vectors_with_concatenate_answer, self.heads_vectors_without_concatenate_answer, self.all_residual_with_concatenate_answer, self.all_residual_without_concatenate_answer = MLPCheck.load_all_data()
            self.intervention_dict_concatenate_answer = self.get_intervention_for_all_data_types(
                self.all_mlp_vector_with_hall_concatenate_answer, self.all_mlp_vector_without_hall_concatenate_answer,
                self.all_attention_vector_with_all_concatenate_answer,
                self.all_attention_vector_without_hall_concatenate_answer, self.all_residual_with_concatenate_answer,
                self.all_residual_without_concatenate_answer, self.heads_vectors_with_concatenate_answer,
                self.heads_vectors_without_concatenate_answer)
            with open(path_intervention_dict_concatenate_answer, 'wb') as handle:
                pickle.dump(self.intervention_dict_concatenate_answer, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # does intervention based on value with concatenation of answer
        self.intervention.all_interventions = self.intervention_dict_concatenate_answer
        self.check_intervention_difference_between_intervention_dicts(self.intervention_dict_concatenate_answer,
                                                                      self.intervention_dict)

    def generate_examples_for_intervention(self, on_test_set):
        """
        generate examples for intervention
        :return:
        """
        train_with_indeces, val_with_indeces, test_with_indeces = self.split_data_to_train_val_test_for_all_data_types(
            [i for i in range(len(self.all_mlp_vector_with_hall))])
        print(f"{len(train_with_indeces)=} {len(val_with_indeces)=} {len(test_with_indeces)=}")
        # assert len(train_with_indeces) + len(val_with_indeces) + len(test_with_indeces) == 1000
        train_with_again, _, _ = self.split_data_to_train_val_test_for_all_data_types([i for i in range(len(self.all_residual_without))])
        assert train_with_again == train_with_indeces, f"{train_with_again} != {train_with_indeces}"
        assert len(train_with_indeces) == len(train_with_again), f"{len(train_with_indeces)} != {len(train_with_again)}"
        assert len(self.hallucinations_examples) >= len(
            self.all_mlp_vector_with_hall), f"{len(self.hallucinations_examples)} != {len(self.all_mlp_vector_with_hall)}"
        self.test_with_examples = [self.hallucinations_examples[i] for i in val_with_indeces]
        self.test_without_examples = [self.non_hallucinations_examples[i] for i in val_with_indeces]
        self.test_general_examples = None if self.general_examples is None else [self.general_examples[i] for i in
                                                                                 val_with_indeces]
        self.mlp_output_of_test_with_examples = [self.all_mlp_vector_with_hall[i] for i in val_with_indeces]
        self.mlp_output_of_test_without_examples = [self.all_mlp_vector_without_hall[i] for i in val_with_indeces]
        self.attention_output_of_test_with_examples = [self.all_attention_vector_with_all[i] for i in val_with_indeces]
        self.attention_output_of_test_without_examples = [self.all_attention_vector_without_hall[i] for i in
                                                          val_with_indeces]
        self.residual_output_of_test_with_examples = [self.all_residual_with[i] for i in val_with_indeces]
        self.residual_output_of_test_without_examples = [self.all_residual_without[i] for i in val_with_indeces]
        if on_test_set:
            print(f"on test set!!")
            self.test_with_examples = [self.hallucinations_examples[i] for i in test_with_indeces]
            self.test_without_examples = [self.non_hallucinations_examples[i] for i in test_with_indeces]
            self.test_general_examples = None if self.general_examples is None else [self.general_examples[i] for i in
                                                                                     test_with_indeces]
        # check that the index of the first example is the same
        assert self.test_without_examples.index(self.test_without_examples[0]) == self.test_with_examples.index(
            self.test_with_examples[
                0]), f"{self.test_without_examples.index(self.test_without_examples[0])} != {self.test_with_examples.index(self.test_with_examples[0])}"
        # count duplicates in each list
        print(
            f"{sum([self.test_without_examples.count(i) for i in self.test_without_examples])-len(self.test_without_examples)=}"
            f"{sum([self.test_with_examples.count(i) for i in self.test_with_examples])-len(self.test_with_examples)=}"
            f"{sum([self.test_general_examples.count(i) for i in self.test_general_examples])-len(self.test_general_examples)=}")

    def check_intervention_difference_between_intervention_dicts(self, intervention_dict1=None,
                                                                 intervention_dict2=None):
        """
        calculate the cosine similarity in mlp,residual,attention between the two intervention dicts
        :return:
        """
        for key in self.intervention_dict.keys():
            print(f"{key=}")
            for layer in self.intervention_dict[key].keys():
                if key == "heads":
                    pass
                else:
                    print(
                        f"{layer=} {round(np.dot(intervention_dict1[key][layer], intervention_dict2[key][layer]) / (np.linalg.norm(intervention_dict2[key][layer]) * np.linalg.norm(intervention_dict1[key][layer])), 2)}")
        return

    def set_what_to_intervene_on(self, use_mlp, use_attention, use_heads, use_residual):
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.intervention.use_mlp = use_mlp
        self.intervention.use_attention = use_attention
        self.intervention.use_heads = use_heads
        self.intervention.use_residual = use_residual

    def load_dataset(self, data_path):
        """
        load the dataset
        :param data_path:
        :return: dataset
        """
        print(f"load dataset {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        print(f"dataset size is {len(data)}")
        # print data hash and its last modification time
        print(f"data name {data_path} and last modification time is {time.ctime(os.path.getmtime(data_path))} ")
        print(f"hash data {subprocess.check_output(['sha1sum', f'{data_path}'])}")
        return data

    def split_data_to_train_val_test(self, data_indexes):
        """
        split data indexes to train val test
        :param data_indexes:
        :return:
        """
        random.seed(42)
        if self.seed_train_val is not None:
            print(f"shuffle all the data with seed {self.seed_train_val}")
            random.seed(self.seed_train_val)
        random.shuffle(data_indexes)
        train_val = data_indexes[:int(len(data_indexes) * 0.8)]
        test = data_indexes[int(len(data_indexes) * 0.8):]
        train = train_val[:int(len(data_indexes) * 0.7)]
        val = train_val[int(len(data_indexes) * 0.7):]
        assert len(train) + len(val) + len(test) == len(data_indexes)
        print(f"{len(train)=} {len(val)=} {len(test)=}")
        return train, val, test

    def split_data_to_train_val_test_for_all_data_types(self, data_split):
        data_indeces = [i for i in range(len(data_split))]
        train_indexes, val_indexes, test_indexes = self.split_data_to_train_val_test(data_indeces)
        train = [data_split[i] for i in train_indexes]
        val = [data_split[i] for i in val_indexes]
        test = [data_split[i] for i in test_indexes]
        return train, val, test

    def linear_classifier(self, train_with, train_without, test_with=None, test_without=None):
        """
        train a linear classifier on the data
        :param train_with: data with hallucinations
        :param train_without: data without hallucinations
        :return: classifier for each layer
        """
        # concatenate the data
        train = train_with + train_without
        labels = [1] * len(train_with) + [0] * len(train_without)
        if test_with is not None and test_without is not None:
            print(f"{len(test_with)=} {len(test_without)=}")
            print(f"{np.shape(np.array(test_with))=} {np.shape(np.array(test_without))=}")
            test = test_with + test_without
            test_labels = [1] * len(test_with) + [0] * len(test_without)
            assert len(test) == len(
                test_labels), f"{len(test)} != {len(test_labels)} {np.shape(np.array(test))=} {np.shape(np.array(test_labels))=}"
            test, test_labels = shuffle(test, test_labels, random_state=0)
        # shuffle the data
        classifier_for_layers = []
        test_acc = []
        test_labels_predicted = []
        train, true_labels = shuffle(train, labels, random_state=0)
        for layer in range(len(train_with[0])):
            train_vectors_curr_layer = np.array([i[layer] for i in train])
            random_state = 0
            if self.seed_train_val is not None:
                random_state = self.seed_train_val
            print(f"random state {random_state}")
            clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
            # use neighbors classifier
            clf.fit(train_vectors_curr_layer, true_labels)
            classifier_for_layers.append(clf)
            if test_with is not None and test_without is not None:
                test_vectors_curr_layer = np.array([i[layer] for i in test])
                test_acc.append(clf.score(test_vectors_curr_layer, test_labels))
                test_labels_predicted.append(clf.predict(test_vectors_curr_layer))
        return classifier_for_layers, test_acc, test_labels_predicted, true_labels

    def linear_classifier_for_heads(self, train_with, train_without, test_with=None, test_without=None):
        """
        train a linear classifier on the data and return the classifier and the test accuracy-for heads
        :param train_with:
        :param train_without:
        :param test_with:
        :param test_without:
        :return:
        """
        # concatenate the data
        train = train_with + train_without
        labels = [1] * len(train_with) + [0] * len(train_without)
        if test_with is not None and test_without is not None:
            print(f"{len(test_with)=} {len(test_without)=}")
            print(f"{np.shape(np.array(test_with))=} {np.shape(np.array(test_without))=}")
            test = test_with + test_without
            test_labels = [1] * len(test_with) + [0] * len(test_without)
            test, test_labels = shuffle(test, test_labels, random_state=0)
        # shuffle the data
        classifier_for_layers = []
        test_acc = []
        train, true_labels = shuffle(train, labels, random_state=0)
        for layer in range(len(train_with[0])):
            layer_classifier = []
            teas_acc_layer = []
            for head in range(len(train_with[0][layer])):
                train_vectors_curr_layer = np.array([i[layer][head] for i in train])
                # print(f"{np.shape(train_vectors_curr_layer)=}")
                random_state = 0
                if self.seed_train_val is not None:
                    random_state = self.seed_train_val
                print(f"random state {random_state}")
                clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
                clf.fit(train_vectors_curr_layer, true_labels)
                layer_classifier.append(clf)
                if test_with is not None and test_without is not None:
                    test_vectors_curr_layer = np.array([i[layer][head] for i in test])
                    teas_acc_layer.append(clf.score(test_vectors_curr_layer, test_labels))
            classifier_for_layers.append(layer_classifier)
            if test_with is not None and test_without is not None:
                test_acc.append(teas_acc_layer)
        print(f"running classifier for heads the result shape is {np.shape(np.array(classifier_for_layers))}")
        return classifier_for_layers, test_acc

    def normClassification(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,4096) - 100 examples,32 layers,4096 features. we want to get the norm of each example and layer (100,32,1)
        train_with_norm_of_each_example_and_layer = np.linalg.norm(train_with, axis=2)
        train_with_norm_of_each_example_and_layer = train_with_norm_of_each_example_and_layer[:, :, np.newaxis]
        print(f"{np.shape(train_with_norm_of_each_example_and_layer)=}")
        train_without_norm_of_each_example_and_layer = np.linalg.norm(train_without, axis=2)
        train_without_norm_of_each_example_and_layer = train_without_norm_of_each_example_and_layer[:, :, np.newaxis]
        test_with_norm_of_each_example_and_layer = np.linalg.norm(test_with, axis=2)
        test_with_norm_of_each_example_and_layer = test_with_norm_of_each_example_and_layer[:, :, np.newaxis]
        test_without_norm_of_each_example_and_layer = np.linalg.norm(test_without, axis=2)
        test_without_norm_of_each_example_and_layer = test_without_norm_of_each_example_and_layer[:, :, np.newaxis]
        print(
            f"{np.shape(test_with_norm_of_each_example_and_layer+test_without_norm_of_each_example_and_layer)=} {np.shape(test_with_norm_of_each_example_and_layer)=} {np.shape(test_without_norm_of_each_example_and_layer)=}")
        classifier_for_layers, test_acc, test_labels_predicted, true_labels = self.linear_classifier(
            list(train_with_norm_of_each_example_and_layer), list(train_without_norm_of_each_example_and_layer),
            list(test_with_norm_of_each_example_and_layer),
            list(test_without_norm_of_each_example_and_layer))
        return test_acc

    def cosineSimilarityClassification(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,4096)
        train_with_normalized = train_with / np.linalg.norm(train_with, axis=2)[:, :, np.newaxis]
        # the norm of each example is 1, check that the norm of each example is 1
        print(f"{np.linalg.norm(train_with_normalized, axis=2)=}")
        for i in train_with_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        print(f"{np.shape(train_with_normalized)=} {np.shape(train_with)=}")
        assert np.shape(train_with_normalized) == np.shape(train_with), f"{np.shape(train_with_normalized)=} {np.shape(train_with)=}"
        train_without_normalized = train_without / np.linalg.norm(train_without, axis=2)[:, :, np.newaxis]
        for i in train_without_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        test_with_normalized = test_with / np.linalg.norm(test_with, axis=2)[:, :, np.newaxis]
        for i in test_with_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        test_without_normalized = test_without / np.linalg.norm(test_without, axis=2)[:, :, np.newaxis]
        for i in test_without_normalized:
            for j in i:
                assert 1.001>np.linalg.norm(j) > 0.95, f"{np.linalg.norm(j)=}"
        classifier_for_layers, test_acc, test_labels_predicted, true_labels = self.linear_classifier(
            list(train_with_normalized), list(train_without_normalized), list(test_with_normalized),
            list(test_without_normalized))
        # assert that the norm of each example is 1
        assert np.linalg.norm(train_with_normalized, axis=2).all() == np.linalg.norm(train_with,
                                                                                     axis=2).all() == np.linalg.norm(
            train_without_normalized, axis=2).all() == np.linalg.norm(train_without,
                                                                      axis=2).all(), f"{np.linalg.norm(train_with_normalized, axis=2)=} {np.linalg.norm(train_with, axis=2)=} {np.linalg.norm(train_without_normalized, axis=2)=} {np.linalg.norm(train_without, axis=2)=}"
        return classifier_for_layers, test_acc, test_labels_predicted, true_labels

    def cosineClassifierForHeads(self, train_with, train_without, test_with, test_without):
        # train with shape is (100,32,32,128)
        train_with_normalized = train_with / np.linalg.norm(train_with, axis=3)[:, :, :, np.newaxis]
        for i in train_with_normalized:
            for j in i:
                for k in j:
                    assert 1.001>np.linalg.norm(k) > 0.95, f"{np.linalg.norm(k)=}"
        train_without_normalized = train_without / np.linalg.norm(train_without, axis=3)[:, :, :, np.newaxis]
        test_with_normalized = test_with / np.linalg.norm(test_with, axis=3)[:, :, :, np.newaxis]
        test_without_normalized = test_without / np.linalg.norm(test_without, axis=3)[:, :, :, np.newaxis]
        assert np.linalg.norm(train_with_normalized, axis=3).all() == np.linalg.norm(train_with,
                                                                                     axis=3).all() == np.linalg.norm(
            train_without_normalized, axis=3).all() == np.linalg.norm(train_without,
                                                                      axis=3).all(), f"{np.linalg.norm(train_with_normalized, axis=3)=} {np.linalg.norm(train_with, axis=3)=} {np.linalg.norm(train_without_normalized, axis=3)=} {np.linalg.norm(train_without, axis=3)=}"
        classifier_for_layers, test_acc = self.linear_classifier_for_heads(list(train_with_normalized),
                                                                           list(train_without_normalized),
                                                                           list(test_with_normalized),
                                                                           list(test_without_normalized))
        return classifier_for_layers, test_acc

    def create_classifier_for_each_type_of_info(self):
        """
        create a classifier for each type of information - mlp,attention,residual,heads
        :return:
        """
        train_mlp_with, val_mlp_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_mlp_vector_with_hall)
        train_mlp_with_again, _, _ = self.split_data_to_train_val_test_for_all_data_types(self.all_mlp_vector_with_hall)
        assert len(train_mlp_with) == len(train_mlp_with_again), f"{len(train_mlp_with)} != {len(train_mlp_with_again)}"
        for i in range(len(train_mlp_with_again)):
            assert torch.eq(torch.tensor(train_mlp_with[i]),torch.tensor(train_mlp_with_again[
                i])).all(), f"{train_mlp_with[i]} != {train_mlp_with_again[i]}"
        train_mlp_without, val_mlp_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_mlp_vector_without_hall)
        classifier_for_layers_mlp, acc_mlp, labels_predicted_mlp, true_labels_mlp = self.cosineSimilarityClassification(
            train_mlp_with, train_mlp_without, val_mlp_with,
            val_mlp_without)
        train_attention_with, val_attention_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_attention_vector_with_all)
        train_attention_without, val_attention_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_attention_vector_without_hall)
        classifier_for_layers_attention, acc_attn, labels_predicted_attn, true_lables_attn = self.cosineSimilarityClassification(
            train_attention_with,
            train_attention_without, val_attention_with,
            val_attention_without)
        train_residual_with, val_residual_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_residual_with)
        train_residual_without, val_residual_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.all_residual_without)
        classifier_for_layers_residual, acc_residual, labels_predicted_residual, true_labels_residual = self.cosineSimilarityClassification(
            train_residual_with,
            train_residual_without, val_residual_with,
            val_residual_without)

        train_heads_with, val_heads_with, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.heads_vectors_with)
        train_heads_without, val_heads_without, _ = self.split_data_to_train_val_test_for_all_data_types(
            self.heads_vectors_without)
        classifier_for_layers_heads, acc_heads = self.cosineClassifierForHeads(train_heads_with,
                                                                               train_heads_without, val_heads_with,
                                                                               val_heads_without)
        # the similarity of the labels predicted
        for l in range(len(labels_predicted_mlp)):
            print(
                f"layer {l} |mlp and attention {accuracy_score(labels_predicted_mlp[l], labels_predicted_attn[l])}| residual and attention {accuracy_score(labels_predicted_residual[l], labels_predicted_attn[l])}| mlp and residual {accuracy_score(labels_predicted_mlp[l], labels_predicted_residual[l])}|mlp acc {acc_mlp[l]}|attn acc {acc_attn[l]}|residual acc {acc_residual[l]}")
        print(
            f"true labels mlp and attention {accuracy_score(true_labels_mlp, true_lables_attn)}| true labels residual and attention {accuracy_score(true_labels_residual, true_lables_attn)}| true labels mlp and residual {accuracy_score(true_labels_mlp, true_labels_residual)}")
        classifier_dict = {"mlp": classifier_for_layers_mlp, "attention": classifier_for_layers_attention,
                           "residual": classifier_for_layers_residual, "heads": classifier_for_layers_heads}
        acc_classifier_dict = {"mlp": acc_mlp, "attention": acc_attn, "residual": acc_residual, "heads": acc_heads}
        if self.seed_train_val is None:
            self.plot_acc_all_types_linear_classifier(acc_classifier_dict)
            self.compare_classifiers(acc_classifier_dict, train_mlp_with, val_mlp_with, train_mlp_without,
                                     val_mlp_without,
                                     train_attention_with, val_attention_with, train_attention_without,
                                     val_attention_without,
                                     train_residual_with, val_residual_with, train_residual_without,
                                     val_residual_without)
        return classifier_dict, acc_classifier_dict




    def compare_classifiers(self, acc_classifier_dict, train_mlp_with, val_mlp_with, train_mlp_without, val_mlp_without,
                            train_attention_with, val_attention_with, train_attention_without, val_attention_without,
                            train_residual_with, val_residual_with, train_residual_without, val_residual_without):
        """
        compare the classifiers, norm,linear and cosine similarity for each type of data
        :param acc_classifier_dict: 
        :param train_mlp_with: 
        :param val_mlp_with: 
        :param train_mlp_without: 
        :param val_mlp_without: 
        :param train_attention_with: 
        :param val_attention_with: 
        :param train_attention_without: 
        :param val_attention_without: 
        :param train_residual_with: 
        :param val_residual_with: 
        :param train_residual_without: 
        :param val_residual_without: 
        :return: 
        """
        print(
            f"{np.shape(train_mlp_with)=} {np.shape(train_mlp_without)=} {np.shape(val_mlp_with)=} {np.shape(val_mlp_without)=}"
            f"{np.shape(train_attention_with)=} {np.shape(train_attention_without)=} {np.shape(val_attention_with)=} {np.shape(val_attention_without)=}"
            f"{np.shape(train_residual_with)=} {np.shape(train_residual_without)=} {np.shape(val_residual_with)=} {np.shape(val_residual_without)=}")
        print(f"compare classifiers")

        _, linear_mlp_acc, _, _ = self.linear_classifier(train_mlp_with, train_mlp_without, val_mlp_with,
                                                         val_mlp_without)
        _, linear_attention_acc, _, _ = self.linear_classifier(train_attention_with, train_attention_without,
                                                               val_attention_with, val_attention_without)
        _, linear_residual_acc, _, _ = self.linear_classifier(train_residual_with, train_residual_without,
                                                              val_residual_with, val_residual_without)

        norm_mlp_acc = self.normClassification(train_mlp_with, train_mlp_without, val_mlp_with, val_mlp_without)
        norm_attention_acc = self.normClassification(train_attention_with, train_attention_without, val_attention_with,
                                                     val_attention_without)
        norm_residual_acc = self.normClassification(train_residual_with, train_residual_without, val_residual_with,
                                                    val_residual_without)

        type_acc_dict = {
            "mlp": {"norm": norm_mlp_acc, "direction": acc_classifier_dict["mlp"], "linear": linear_mlp_acc},
            "attention": {"norm": norm_attention_acc, "direction": acc_classifier_dict["attention"],
                          "linear": linear_attention_acc},
            "residual": {"norm": norm_residual_acc, "direction": acc_classifier_dict["residual"],
                         "linear": linear_residual_acc}}
        self.plot_acc_all_types_compare_classifier(type_acc_dict)

    def plot_acc_all_types_compare_classifier(self, acc_classifier_dict):
        """
        plot the accuracy per type using all classifiers
        :param acc_classifier_dict:
        :return:
        """
        print(f"plot acc all types compare classifier")
        for type, acc_dict in acc_classifier_dict.items():
            plt.figure()
            plt.title(f"Classifiers accuracy on the {type}",fontsize=20)
            plt.xlabel("layer",fontsize=15)
            plt.ylabel("accuracy",fontsize=15)
            plt.ylim(40, 100)
            colors = ["blue", "orange", "red"]
            line_dash = ["solid", "dashed", "dotted"]
            key_color_dict = {"direction": ["blue", "solid"], "norm": ["purple", "dashed"], "linear": ["red", "dotted"]}
            for key in acc_dict.keys():
                plt.plot([100*val for val in acc_dict[key]] , label=key, color=key_color_dict[key][0],
                         linestyle=key_color_dict[key][1])
            plt.legend(fontsize=15)
            plt.grid()
            plt.savefig(f"{self.path_to_save_results}accuracy_{type}_compare_classifier_on_val.pdf", format="pdf")
            plt.close()

    def plot_acc_all_types_linear_classifier(self, acc_classifier_dict):
        """
        plot the accuracy per classifier
        :param acc_per_classifier_mlp:
        :param label:
        :return:
        """
        plt.figure()
        plt.title(f"Classifier accuracy", fontsize=20)
        plt.xlabel("layer", fontsize=15)
        plt.ylabel("accuracy", fontsize=15)
        plt.ylim(40, 100)
        colors = ["green", "red", "blue"]
        line_dash = ["solid", "dashed", "dotted"]
        key_color_dict = {"mlp": ["red", "dashed"], "residual": ["green", "solid"], "attention": ["blue", "dotted"]}
        for i, key in enumerate(acc_classifier_dict.keys()):
            if key != "heads":
                plt.plot([val * 100 for val in acc_classifier_dict[key]], label=key, color=key_color_dict[key][0],
                         linestyle=key_color_dict[key][1])
        plt.legend(fontsize=15)
        plt.grid()
        plt.savefig(f"{self.path_to_save_results}accuracy_linear_classifier_on_val.pdf", format="pdf")
        plt.close()

    def get_intervention(self, label_with_data, label_without_data):
        """
        get the intervention for each layer - the difference between the mean of the data with hallucinations and the mean of the data without hallucinations
        :param label_with_data:
        :param label_without_data:
        :return: a dict with the intervention for each layer
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            assert layer < 50
            print(
                f"{layer=}|{np.dot(np.mean(label_with_data, axis=0)[layer], np.mean(label_without_data, axis=0)[layer]) / (np.linalg.norm(np.mean(label_without_data, axis=0)[layer]) * np.linalg.norm(np.mean(label_with_data, axis=0)[layer]))} | {np.mean(np.mean(label_with_data, axis=0)[layer])}| {np.mean(np.std(label_with_data, axis=0)[layer])} |{np.mean(np.mean(label_without_data, axis=0)[layer])}| {np.mean(np.std(label_without_data, axis=0)[layer])}")
            print(
                f"std of the vectors with {np.mean(np.std(label_with_data, axis=0)[layer])} std of the vectors without {np.mean(np.std(label_without_data, axis=0)[layer])}")
            print(
                f"norm with {np.linalg.norm(np.mean(label_with_data, axis=0)[layer])} norm without {np.linalg.norm(np.mean(label_without_data, axis=0)[layer])} norm without minus with {np.linalg.norm(np.mean(label_without_data, axis=0)[layer] - np.mean(label_with_data, axis=0)[layer])}")

            layer_intervention_val[layer] = np.mean(label_with_data, axis=0)[layer] - \
                                            np.mean(label_without_data, axis=0)[layer]
            assert np.shape(layer_intervention_val[layer]) == np.shape(label_with_data[0][layer])
        return layer_intervention_val

    def get_intervention_based_classifier(self, label_with_data, label_without_data, type="mlp"):
        """
        use self.classifiers_dict to get the classifier plane and take the perpendicular vector to the direction of without data to be the intervention
        :param label_with_data:
        :param label_without_data:
        :param classifier_for_layers:
        :return:
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            classifier = self.classifiers_dict[type][layer]
            # the hyperplane of the linearsvc classifier
            w = classifier.coef_[0]
            # the perpendicular vector to the hyperplane that is in the direction of the without data
            d = w / np.linalg.norm(w)
            # check that the direction of the without data is in the direction of the without data
            mass_mean_vector = np.mean(label_with_data, axis=0)[layer] - np.mean(label_without_data, axis=0)[layer]
            print(
                f"layer {layer} cosine similarity d and classifier d {np.dot(d, mass_mean_vector) / (np.linalg.norm(d) * np.linalg.norm(mass_mean_vector))}")
            print(f"classifier d norm {np.linalg.norm(d)} mass_mean_vector norm {np.linalg.norm(mass_mean_vector)}")
            if np.dot(d, mass_mean_vector) < 0:
                print(f"layer {layer} is in the direction of the without data")
                d = -d
            # normalize the vector to be in the same norm as the mass_mean_vector
            d = d * np.linalg.norm(mass_mean_vector)
            assert abs(np.linalg.norm(d) - np.linalg.norm(mass_mean_vector))<0.001, f"{np.linalg.norm(d)} != {np.linalg.norm(mass_mean_vector)}"
            layer_intervention_val[layer] = d
        return layer_intervention_val

    def get_intervention_based_classifier_for_heads(self, label_with_data, label_without_data, type="heads"):
        """
        use self.classifiers_dict to get the classifier plane and take the perpendicular vector to the direction of without data to be the intervention
        :param label_with_data:
        :param label_without_data:
        :param classifier_for_layers:
        :param type:
        :return:
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            layer_intervention_val[layer] = {}
            for head in range(len(label_with_data[0][layer])):
                assert layer < 50 and head < 40
                assert type in self.classifiers_dict.keys()
                assert type == "heads"
                classifier = self.classifiers_dict[type][layer][head]
                w = classifier.coef_[0]
                d = w / np.linalg.norm(w)
                mass_mean_vector = np.mean(label_with_data, axis=0)[layer][head] - \
                                   np.mean(label_without_data, axis=0)[layer][head]
                print(
                    f"layer {layer} head {head} cosine similarity d and classifier d {np.dot(d, mass_mean_vector) / (np.linalg.norm(d) * np.linalg.norm(mass_mean_vector))}")
                print(f"classifier d norm {np.linalg.norm(d)} mass_mean_vector norm {np.linalg.norm(mass_mean_vector)}")
                if np.dot(d, mass_mean_vector) < 0:
                    print(f"layer {layer} head {head} is in the direction of the without data")
                    d = -d
                d = d * np.linalg.norm(mass_mean_vector)
                assert abs(np.linalg.norm(d) - np.linalg.norm(mass_mean_vector))<0.001, f"{np.linalg.norm(d)} != {np.linalg.norm(mass_mean_vector)}"
                layer_intervention_val[layer][head] = d
        return layer_intervention_val

    def get_intervention_for_heads(self, label_with_data, label_without_data):
        """
        get the intervention for each layer and head - the difference between the mean of the data with hallucinations and the mean of the data without hallucinations
        :param label_with_data:
        :param label_without_data:
        :return: a dict with the intervention for each layer and head
        """
        layer_intervention_val = {}
        for layer in range(len(label_with_data[0])):
            layer_intervention_val[layer] = {}
            for head in range(len(label_with_data[0][layer])):
                assert layer < 50 and head < 40
                print(
                    f"head={layer},{head} norm with {np.linalg.norm(np.mean(label_with_data, axis=0)[layer][head])} norm without {np.linalg.norm(np.mean(label_without_data, axis=0)[layer][head])} norm without minus with {np.linalg.norm(np.mean(label_without_data, axis=0)[layer][head] - np.mean(label_with_data, axis=0)[layer][head])}")

                # print(
                #     f"{layer=}|{head=}| {np.mean(np.mean(label_with_data, axis=0)[layer][head])}| {np.mean(np.std(label_with_data, axis=0)[layer][head])} |{np.mean(np.mean(label_without_data, axis=0)[layer][head])}| {np.mean(np.std(label_without_data, axis=0)[layer][head])}")
                layer_intervention_val[layer][head] = np.mean(label_with_data, axis=0)[layer][head] - \
                                                      np.mean(label_without_data, axis=0)[layer][head]

                assert np.shape(layer_intervention_val[layer][head]) == np.shape(label_with_data[0][layer][head])
        return layer_intervention_val

    def get_intervention_for_all_data_types_based_classifier(self, all_mlp_vector_with_hall,
                                                             all_mlp_vector_without_hall,
                                                             all_attention_vector_with_all,
                                                             all_attention_vector_without_hall,
                                                             all_residual_with, all_residual_without,
                                                             heads_vectors_with=None,
                                                             heads_vectors_without=None):
        train_mlp_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        train_mlp_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_without_hall)
        assert len(train_mlp_with) == len(train_mlp_without), f"{len(train_mlp_with)} != {len(train_mlp_without)}"
        print(f"MLP direction for intervention")
        intervention_mlp = self.get_intervention_based_classifier(train_mlp_with, train_mlp_without, type="mlp")
        train_attention_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_with_all)
        train_attention_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_without_hall)
        assert len(train_attention_with) == len(
            train_attention_without), f"{len(train_attention_with)} != {len(train_attention_without)}"
        print(f"Attention direction for intervention")
        intervention_attention = self.get_intervention_based_classifier(train_attention_with, train_attention_without,
                                                                        type="attention")
        train_residual_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_with)
        train_residual_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_without)
        assert len(train_residual_with) == len(
            train_residual_without), f"{len(train_residual_with)} != {len(train_residual_without)}"
        print(f"Residual direction for intervention")
        intervention_residual = self.get_intervention_based_classifier(train_residual_with, train_residual_without,
                                                                       type="residual")
        train_heads_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_with)
        train_heads_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_without)
        assert len(train_heads_with) == len(
            train_heads_without), f"{len(train_heads_with)} != {len(train_heads_without)}"
        print(f"Heads direction for intervention")
        intervention_heads = self.get_intervention_based_classifier_for_heads(train_heads_with, train_heads_without,
                                                                              type="heads")
        intervention_dict = {"mlp": intervention_mlp, "attention": intervention_attention,
                             "residual": intervention_residual, "heads": intervention_heads}
        return intervention_dict

    def get_intervention_for_all_data_types(self, all_mlp_vector_with_hall, all_mlp_vector_without_hall,
                                            all_attention_vector_with_all, all_attention_vector_without_hall,
                                            all_residual_with, all_residual_without, heads_vectors_with=None,
                                            heads_vectors_without=None):
        """
        get the intervention for each type of data - mlp,attention,residual,heads, using the train data
        :param all_mlp_vector_with_hall:
        :param all_mlp_vector_without_hall:
        :param all_attention_vector_with_all:
        :param all_attention_vector_without_hall:
        :param all_residual_with:
        :param all_residual_without:
        :param heads_vectors_with:
        :param heads_vectors_without:
        :return:
        """
        train_mlp_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        train_mlp_with2, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall)
        for i in range(len(train_mlp_with)):
            assert torch.eq(torch.tensor(train_mlp_with[i]), torch.tensor(train_mlp_with2[i])).all(), f"{train_mlp_with[i]} != {train_mlp_with2[i]}"
        train_mlp_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_mlp_vector_without_hall)
        assert len(train_mlp_with) == len(train_mlp_without), f"{len(train_mlp_with)} != {len(train_mlp_without)}"
        print(f"MLP direction for intervention")
        intervention_mlp = self.get_intervention(train_mlp_with, train_mlp_without)
        train_attention_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_with_all)
        train_attention_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(
            all_attention_vector_without_hall)
        assert len(train_attention_with) == len(
            train_attention_without), f"{len(train_attention_with)} != {len(train_attention_without)}"
        print(f"Attention direction for intervention")
        intervention_attention = self.get_intervention(train_attention_with, train_attention_without)
        train_residual_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_with)
        train_residual_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(all_residual_without)
        assert len(train_residual_with) == len(
            train_residual_without), f"{len(train_residual_with)} != {len(train_residual_without)}"
        print(f"Residual direction for intervention")
        intervention_residual = self.get_intervention(train_residual_with, train_residual_without)
        train_heads_with, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_with)
        train_heads_without, _, _ = self.split_data_to_train_val_test_for_all_data_types(heads_vectors_without)
        assert len(train_heads_with) == len(
            train_heads_without), f"{len(train_heads_with)} != {len(train_heads_without)}"
        print(f"Heads direction for intervention")
        intervention_heads = self.get_intervention_for_heads(train_heads_with, train_heads_without)
        intervention_dict = {"mlp": intervention_mlp, "attention": intervention_attention,
                             "residual": intervention_residual, "heads": intervention_heads}
        return intervention_dict


class ModelIntervention():
    def __init__(self, path_to_results, model_name,
                 dataset_size, dataset_name, threshold_of_data, use_mlp, use_attention, use_heads, use_residual,
                 intervention_all_dict, classifier_dict,
                 alpha=5, normalize=False, acc_classification_val=None, static_intervention=False, check_mlp_out=None,
                 check_attention_out=None, check_residual_out=None):
        self.check_mlp_out = check_mlp_out
        self.check_attention_out = check_attention_out
        self.check_residual_out = check_residual_out
        self.static_intervention = static_intervention
        self.static_random_intervention = False
        self.last_hook = ""
        self.alpha = alpha
        self.normalize = normalize
        self.use_mlp = use_mlp
        self.use_attention = use_attention
        self.use_heads = use_heads
        self.use_residual = use_residual
        self.path_to_save_results = path_to_results + f"{model_name.replace('/', '_')}" + f'{"/"}' + f"{dataset_name}/{threshold_of_data}/concat_answer{False}_size{dataset_size}/"
        self.all_interventions = intervention_all_dict
        # print(f"{self.all_interventions=}")
        self.classifier_dict = classifier_dict
        self.acc_classification_val = acc_classification_val
        self.best_layers_by_val_acc = {"mlp": np.argsort(self.acc_classification_val["mlp"])[-5:],
                                       "attention": np.argsort(
                                           self.acc_classification_val["attention"])[-5:], "residual": np.argsort(
                self.acc_classification_val["residual"])[-5:]}
        # sort by acc heads- list of tuples (layer,head) sorted by acc
        self.best_layers_by_val_acc["heads"] = sorted(
            [(layer, head) for layer in range(len(self.acc_classification_val["heads"])) for head in
             range(len(self.acc_classification_val["heads"][layer]))],
            key=lambda x: self.acc_classification_val["heads"][x[0]][x[1]])[-50:]
        np.random.seed(0)

        print(f"{self.best_layers_by_val_acc=}")
        self.min_acc = 0.65
        self.number_of_interventions = 0
        print(f"{self.min_acc=}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        transformers.GenerationConfig.do_sample = False
        transformers.GenerationConfig.top_p = 0
        transformers.GenerationConfig.temperature = None
        # print(f"model name {model_name} hash is {self.model.config._commit_hash}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            print(f"Setting pad_token to eos_token: {self.tok.pad_token}")
        model_creator = get_model_config()
        self.model_condig = model_creator.model_config(model_name, self.model)
        self.MODEL_NAME = self.model_condig["model_type"]

        self.call_hook_times = 0

    def rgetattr(self, obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

        # a safe way to set attribute of an object

    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)

    def wrap_model(self, model,
                   layers_to_check: List[str],
                   max_len=1000):

        hs_collector = {}
        handles = []
        options = {"." + self.model_condig["mlp_name"]: self.use_mlp,
                   "." + self.model_condig["attention_name"]: self.use_attention,
                   "." + self.model_condig["attn_proj_name"]: self.use_heads,
                   "": self.use_residual}
        final_layers_to_check = [layer_type for layer_type in layers_to_check if options[layer_type]]
        print(f"{final_layers_to_check=}")
        layers_to_check = final_layers_to_check
        for layer_idx in range(self.model_condig["num_hidden_layers"]):
            for layer_type in layers_to_check:
                layer_with_idx = f'{layer_idx}{layer_type}'
                inside_name = f"{self.model_condig['start_layer_prefex']}{layer_with_idx}"
                layer_pointer = self.rgetattr(model, inside_name)
                add_handler = True
                if add_handler:
                    # print(f"adding handler for {layer_with_idx}")
                    handel = layer_pointer.register_forward_hook(self.changeActivationOutput(
                        layer_i=layer_idx,
                        layer_type=layer_type
                    ))
                    handles.append(handel)

        return hs_collector, handles

    def attention_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the attention layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = (new_output[0].clone().detach().to(device),) + output[1:]
        # print(f"{self.all_interventions['attention'][layer_i]=}")
        new_output[0][0][-1] = output[0][0][-1] - alpha * torch.tensor(
            self.all_interventions['attention'][layer_i]).squeeze(0).to(device)
        if not self.use_attention:
            new_output = output
        return new_output

    def mlp_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the mlp layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = output.clone().detach().to(device)
        new_output[0][-1] = output[0][-1] - alpha * torch.tensor(self.all_interventions['mlp'][layer_i]).squeeze(
            0).to(device)
        assert new_output.shape == output.shape, f"{new_output.shape=} {output.shape=}"
        if not self.use_mlp:
            new_output = output
        return new_output

    def residual_intervention(self, layer_i, layer_type, output, alpha):
        """
        change the output of the residual layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        new_output = output
        new_output = (new_output[0].clone().detach().to(device),) + output[1:]
        new_output[0][0][-1] = output[0][0][-1] - alpha * torch.tensor(
            self.all_interventions['residual'][layer_i]).squeeze(0).to(device)
        assert new_output[0].shape == output[0].shape, f"{new_output[0].shape=} {output[0].shape=}"
        if not self.use_residual:
            new_output = output
        return new_output

    def heads_intervention(self, layer_i, layer_type, output, alpha, input_, module):
        """
        change the output of the heads layer
        :param layer_i:
        :param layer_type:
        :param output:
        :param alpha:
        :param With_norm:
        :return:
        """
        dim_head = self.model_condig["hidden_size"] // self.model_condig["num_attention_heads"]
        new_input = input_
        old_input = input_[0].clone().detach().to(device)
        heads_interventions = 0
        for head in range(self.model_condig["num_attention_heads"]):
            clf = self.classifier_dict['heads'][layer_i][head]
            vector_head = old_input[0][-1][dim_head * head:dim_head * (head + 1)].detach().cpu().numpy()
            predict_val_norm_vector = clf.predict([vector_head / np.linalg.norm(vector_head)])
            assert 1.001>np.linalg.norm(vector_head/np.linalg.norm(vector_head)) > 0.95, f"{np.linalg.norm(vector_head/np.linalg.norm(vector_head))=}"
            non_static_intervention = self.static_intervention == False and \
                                      self.acc_classification_val['heads'][layer_i][
                                          head] > self.min_acc and predict_val_norm_vector == 1
            static_intervention = self.static_intervention == True and (layer_i, head) in self.best_layers_by_val_acc[
                "heads"]
            if non_static_intervention or static_intervention:
                self.number_of_interventions += 1
                heads_interventions += 1
                val = self.all_interventions['heads'][layer_i][head]
                new_input[0][0][-1][dim_head * head:dim_head * (head + 1)] -= alpha * torch.tensor(
                    val).squeeze(
                    0).to(device)
        if heads_interventions > 0:
            # assert that not all values in new_input are the same as the old input
            assert not torch.eq(new_input[0][0][-1], old_input[0][-1]).all(), f"{new_input[0][0][-1]=} {old_input[0][-1]=}"
            new_output = module.forward(*new_input)
            # print(f"intervention on layer {layer_i} heads {heads_interventions} times")
        else:
            assert torch.eq(new_input[0][0][-1], old_input[0][-1]).all(), f"{new_input[0][0][-1]=} {old_input[0][-1]=}"
            new_output = output
        assert torch.all(
            torch.eq(output[0], module.forward(*(old_input,))[0])), f"{output=} {module.forward(*input_)=}"
        assert np.shape(new_output) == np.shape(output), f"{np.shape(new_output)=} {np.shape(output)=}"
        return new_output

    def changeActivationOutput(self, layer_i, layer_type):
        def hook(module, input_, output):
            self.call_hook_times += 1
            alpha = self.alpha
            new_output = output
            next_hook = str(layer_i) + "_" + layer_type
            # print(f"{next_hook=}, {self.last_hook=}")
            assert self.last_hook != next_hook, f"{self.last_hook=} {next_hook=}"
            self.last_hook = next_hook
            # attention intervention
            if layer_type == "." + self.model_condig["attention_name"] and self.use_attention is True:
                clf = self.classifier_dict['attention'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["attention"]
                if static:
                    print(f"intervention on attention layer {layer_i} static")
                vector_attention = output[0][0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_attention / np.linalg.norm(vector_attention)])
                assert 1.001>np.linalg.norm(vector_attention/np.linalg.norm(vector_attention)) > 0.95, f"{np.linalg.norm(vector_attention/np.linalg.norm(vector_attention))=}"
                non_static = self.static_intervention == False and self.acc_classification_val['attention'][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    # print(f"intervention on attention layer {layer_i}")
                    self.number_of_interventions += 1
                    new_output = self.attention_intervention(layer_i, layer_type, output, alpha)
                    # print(f"intervention on attention layer {layer_i}")
                if static:
                    print(
                        f"The similarity before and after the intervention is {torch.cosine_similarity(output[0][0][-1], new_output[0][0][-1], dim=0)}")
            # mlp intervention
            elif layer_type == "." + self.model_condig["mlp_name"] and self.use_mlp is True:
                clf = self.classifier_dict['mlp'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["mlp"]
                if static:
                    print(f"intervention on mlp layer {layer_i} static")
                vector_mlp = output[0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_mlp / np.linalg.norm(vector_mlp)])
                assert 1.001>np.linalg.norm(vector_mlp/np.linalg.norm(vector_mlp)) > 0.95, f"{np.linalg.norm(vector_mlp/np.linalg.norm(vector_mlp))=}"
                non_static = self.static_intervention == False and self.acc_classification_val["mlp"][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    # print(f"intervention on mlp layer {layer_i}")
                    self.number_of_interventions += 1
                    new_output = self.mlp_intervention(layer_i, layer_type, output, alpha)
                if static:
                    print(
                        f"The similarity before and after the intervention is {torch.cosine_similarity(output[0][-1], new_output[0][-1], dim=0)}")
            elif layer_type == "" and self.use_residual is True:
                clf = self.classifier_dict['residual'][layer_i]
                static = self.static_intervention == True and layer_i in self.best_layers_by_val_acc["residual"]
                if static:
                    print(f"intervention on residual layer {layer_i} static")
                vector_residual = output[0][0][-1].detach().cpu().numpy()
                predict_val_norm_vector = clf.predict(
                    [vector_residual / np.linalg.norm(vector_residual)])
                assert 1.001>np.linalg.norm(vector_residual/np.linalg.norm(vector_residual)) > 0.95, f"{np.linalg.norm(vector_residual/np.linalg.norm(vector_residual))=}"
                non_static = self.static_intervention == False and self.acc_classification_val['residual'][
                    layer_i] > self.min_acc and predict_val_norm_vector == 1
                if static or non_static:
                    self.number_of_interventions += 1
                    # print(f"intervention on residual layer {layer_i}")
                    new_output = self.residual_intervention(layer_i, layer_type, output, alpha)
                if static:
                    print(
                        f"The similarity before and after the intervention is {torch.cosine_similarity(output[0][0][-1], new_output[0][0][-1], dim=0)}")
            elif layer_type == "." + self.model_condig["attn_proj_name"] and self.use_heads is True:
                new_output = self.heads_intervention(layer_i=layer_i, layer_type=layer_type, output=output, alpha=alpha,
                                                     input_=input_, module=module)

            assert new_output[0].shape == output[0].shape, f"{new_output[0].shape=} {output[0].shape=}"
            return new_output

        return hook

    def model_wrap_remove_hooks(self, model, handels_to_remove: List[torch.utils.hooks.RemovableHandle]):
        """
        remove hooks from model
        :param model:
        :param handels_to_remove:
        :return:
        """
        for handel in handels_to_remove:
            handel.remove()

    def run_model_with_hook(self, model, input_encoded):
        """
        run model with hook that will change values after some MLP and attention layers
        :param model:
        :param input_encoded:
        :return: model's output
        """
        model.requires_grad_(False)
        self.call_hook_times = 0
        # collect the hidden states before and after each of those layers (modules)
        hs_collector, handles = self.wrap_model(model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                        "." + self.model_condig["attention_name"],
                                                                        "." + self.model_condig["attn_proj_name"], ""])
        with torch.no_grad():
            output = model(input_encoded, output_hidden_states=True, output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
        gc.collect()
        assert self.call_hook_times % 32 == 0, f"{self.call_hook_times=}"
        assert self.call_hook_times <= 32 , f"{self.call_hook_times=}"
        # remove hooks
        self.model_wrap_remove_hooks(model, handles)

        return output

    def run_model_generate_with_hook(self, model, input_encoded, length=generation_length,
                                     min_length=generation_length):

        start_time = time.time()
        model.requires_grad_(False)
        hs_collector, handles = self.wrap_model(model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                        "." + self.model_condig["attention_name"],
                                                                        "." + self.model_condig["attn_proj_name"], ""])
        with torch.no_grad():
            # if type(input_encoded) == torch.Tensor:
            self.call_hook_times = 0
            print(f"{len(input_encoded[0])=}")
            output = model.generate(input_encoded, output_hidden_states=True, output_attentions=True, use_cache=False,
                                    max_length=(len(input_encoded[0]) + length), do_sample=False,
                                    num_beams=1, attention_mask=torch.ones(input_encoded.shape).to(device),
                                    pad_token_id=self.tok.eos_token_id, min_new_tokens=min_length)
            print(f"number of times the hook was called is {self.call_hook_times}")
            assert self.call_hook_times % 32 == 0, f"{self.call_hook_times=}"
        assert len(output[0]) <= len(input_encoded[0]) + length
        assert len(output[0]) >= len(input_encoded[0]) + min_length, f"{len(output[0])=} {len(input_encoded[0])=}"
        # print(f"{output[0]=} {len(output[0])=} {len(input_encoded[0])=}")
        generated = self.tok.batch_decode(output, skip_special_tokens=True)
        # remove hooks
        self.model_wrap_remove_hooks(model, handles)
        # print the time it took to generate
        print(f"took {time.time() - start_time} seconds to generate")
        return generated

    def get_answer_rank(self, model, prompt, answer_tokens):
        """
        get the mean rank if the two answers are the same else get the difference between the the first different token
        between the two answers
        :param model:
        :param prompt:
        :param answer_tokens:
        :return: if less than zero prefer the new answer and if more than zero prefer the old answer
        """
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        # print(f"{encoded_line=} {encoded_line.shape=}")
        model_out = self.run_model_with_hook(model, input_ids)
        self.run_mlp_check = False
        self.run_residual_check = False
        self.run_attention_check = False
        logits = model_out.logits
        probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        # probability of the first token of the old and new answer
        prob_old = [probabilities[0][answer_tokens[0][0]].item()]
        prob_new = [probabilities[0][answer_tokens[1][0]].item()]
        rank_old = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[0]])]
        rank_new = [sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[0]])]
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        for i, token in enumerate(answer_tokens[0][:-1]):
            # add the token to the input_ids
            next_token = answer_tokens[0][i + 1]
            input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
            model_out = self.run_model_with_hook(model, input_ids)
            logits = model_out.logits
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            prob_old.append(probabilities[0][next_token].item())
            rank_old.append(sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_old[-1]]))

        assert len(prob_old) == len(answer_tokens[0]), f"{len(prob_old)=} {len(answer_tokens[0])=}"
        # print(f"{prob_old=}")
        input_ids = self.tok([prompt], return_tensors="pt")["input_ids"].to(device)
        if answer_tokens[0] == answer_tokens[1]:
            prob_new = prob_old
        else:
            for i, token in enumerate(answer_tokens[1][:-1]):
                next_token = answer_tokens[1][i + 1]
                input_ids = torch.cat([input_ids, torch.tensor([[token]]).to(device)], dim=-1)
                model_out = self.run_model_with_hook(model, input_ids)
                logits = model_out.logits
                probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                prob_new.append(probabilities[0][next_token].item())
                rank_new.append(
                    sum([1 for i in range(len(probabilities[0])) if probabilities[0][i].item() > prob_new[-1]]))

        assert len(prob_new) == len(answer_tokens[1]), f"{len(prob_new)=} {len(answer_tokens[1])=}"
        # geometric mean
        prob_old = np.prod(np.array(prob_old)) ** (1 / len(prob_old))
        prob_new = np.prod(np.array(prob_new)) ** (1 / len(prob_new))
        shorter_answer = min(len(answer_tokens[0]), len(answer_tokens[1]))
        non_similar_index = [i for i in range(shorter_answer) if answer_tokens[0][i] != answer_tokens[1][i]]
        print(f"{non_similar_index=} {answer_tokens=}")
        if len(non_similar_index) > 0:
            assert answer_tokens[0][non_similar_index[0]] != answer_tokens[1][non_similar_index[
                0]], f"{answer_tokens[0][non_similar_index[0]]=} {answer_tokens[1][non_similar_index[0]]=}"
        if len(non_similar_index) == 0:
            non_similar_index = [0]
        rank = rank_new[non_similar_index[0]] - rank_old[non_similar_index[0]]
        if answer_tokens[0] == answer_tokens[1]: #no context setting
            print(f"{prob_old=}, {prob_new=}")
            return prob_old, sum(rank_old) / len(rank_old)
        del input_ids
        del model_out
        torch.cuda.empty_cache()
        return prob_old - prob_new, rank



    def run_dataset_with_hook(self, dataset, tag, check_mlp_out=None, check_attention_out=None, check_residual_out=None,
                              no_context_dataset=False, calculate_wiki_pp=False):
        """
        run the dataset with the hook, it will check the dataset results with modification to the model inner state (the hook)
        It will print the std and mean of the probability of parametric answer - the probability of the contextual answer
        :param dataset_path:
        :param tag:
        :return:
        """
        print(f"{self.min_acc=}")
        print(f"{no_context_dataset=}")
        print(f"{self.use_attention=} {self.use_mlp=} {self.use_residual=} {self.use_heads=}")
        self.check_mlp_out = check_mlp_out
        self.check_attention_out = check_attention_out
        self.check_residual_out = check_residual_out
        print(np.shape(self.check_mlp_out))
        self.number_of_interventions = 0
        self.wanted_intervention = 0
        self.wanted_mlp_intervention = 0
        self.wanted_attention_intervention = 0
        self.wanted_residual_intervention = 0
        rank_bigger_that_zero = 0
        no_context_rank_high = 0
        non_text_generated = 0
        generated_both_answers = 0
        prompt_index = 0
        parametric_answer = 1
        contextual_answer = 2
        paraphraze_prompt_index = 5
        logits_on_true_answer_without_context_index = -1
        number_of_examples_used = 0
        preferable_answer_prob = []
        preferable_answer_rank = []
        generated_preferable_answer = []
        generated_text = []
        for index, point in enumerate(dataset):
            self.index = index
            self.run_mlp_check = True
            self.run_residual_check = True
            self.run_attention_check = True
            if index % 10 == 0:
                print(f"index is {index} with {number_of_examples_used=}", flush=True)
                assert number_of_examples_used == index
            if index < 5:
                print(f"{point=}")
            number_of_examples_used += 1

            prompt = point[prompt_index]
            old_target = point[1]
            new_target = point[2]
            answer_tokens = (point[3], point[4])
            if no_context_dataset:
                assert answer_tokens[0] == answer_tokens[1], f"{answer_tokens=}"
                assert old_target == new_target, f"{old_target=} {new_target=}"
            praphraze_prompt = point[paraphraze_prompt_index]
            start_time = time.time()
            prob, rank = self.get_answer_rank(self.model, praphraze_prompt, answer_tokens)
            if no_context_dataset:
                new_prompt = self.add_good_shots(prompt, praphraze_prompt)
                print(f"{new_prompt=} {praphraze_prompt=}")
                no_context_prob, no_context_rank = self.get_answer_rank(self.model, new_prompt,
                                                                        answer_tokens)
                prob = -prob
                rank = rank - no_context_rank
            else:
                if rank == 0:
                    print(f"with context the rank is zero {rank=}")
            preferable_answer_prob.append(prob)
            preferable_answer_rank.append(rank)
            print(f"took {time.time() - start_time} seconds to get the probability difference")
            generated = self.run_model_generate_with_hook(self.model, self.tok([praphraze_prompt], return_tensors="pt")[
                "input_ids"].to(device))
            if index < 70:
                print(
                    f'{generated=}')
            generated_text.append((generated[0], generated[0][len(praphraze_prompt):], old_target, new_target))
            generated = generated[0][len(praphraze_prompt):]
            print(f"{generated=}")
            if len(self.tok(generated, return_tensors="pt")["input_ids"][0]) < generation_length:
                print(f"short generation {len(self.tok(generated, return_tensors='pt')['input_ids'][0])=} {generated=}")
            if len(generated) == 0:
                non_text_generated += 1
            wanted_answer_generated = 0
            if (
                    old_target.strip() in generated or old_target.strip().lower() in generated.lower()) and not no_context_dataset:
                wanted_answer_generated += 1
            if (new_target.strip() in generated or new_target.strip().lower() in generated.lower()):
                wanted_answer_generated -= 1
            if old_target in generated and new_target in generated:
                generated_both_answers += 1
            generated_preferable_answer.append(wanted_answer_generated)

        if calculate_wiki_pp:
            pp_wikipedia, generated_wiki = self.get_wikipedia_pp_score()
        else:
            pp_wikipedia = None
            generated_wiki = None

        perplexity = None
        return preferable_answer_rank, generated_preferable_answer, perplexity, generated_text, pp_wikipedia, generated_wiki




    def get_wikipedia_pp_score(self):
        """
        get the perplexity of wikipedia text using the model with the hook
        :return:
        """
        print(f"start wikipedia pp score", flush=True)
        start_time = time.time()
        np.random.seed(42)

        def calc_loss(labels, logits):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss
        if not os.path.exists("wiki_dataset"):
            # save the wikipedia dataset
            wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", ignore_verifications=True)
            wiki.save_to_disk("wiki_dataset")
        data_wiki = load_from_disk("wiki_dataset")
        indexes_of_title = [i for i in range(len(data_wiki)) if
                            "=" in data_wiki[i]["text"] and data_wiki[i][
                                "text"].count("=") == 2]
        print(f"{len(indexes_of_title)=}", flush=True)
        wanted_indexes = []
        for i in indexes_of_title:
            article = ""
            last_paragraph = ""
            article += data_wiki[i]["text"]
            j = 1
            count = 0
            while i + j < len(data_wiki) and data_wiki[i + j]["text"].count("=") != 2:
                if count < 2 and data_wiki[i + j]["text"] != "":
                    article += data_wiki[i + j]["text"]
                    count += 1
                j += 1
                if count == 2:
                    wanted_indexes.append((article, last_paragraph))
                    break
                if j > 100:
                    print(f"more than 100 paragraphs past j is {j} and i is {i} text is {data_wiki[i]['text']}")
                    break

        sample = [wanted_indexes[i] for i in np.random.choice(len(wanted_indexes), 100, replace=False)]
        print(sample[:3])
        del data_wiki
        sample = sample[:100]
        print(f"{len(sample)=}")
        assert len(sample) == 100, f"{len(sample)=}"
        wikipedia_pp_score = []
        direct_pp_score_no_intervention = []
        texts = []
        list_loss = []

        # collect the hidden states before and after each of those layers (modules)
        hs_collector, handles = self.wrap_model(self.model, layers_to_check=["." + self.model_condig["mlp_name"],
                                                                             "." + self.model_condig["attention_name"],
                                                                             "." + self.model_condig["attn_proj_name"],
                                                                             ""])

        for index, s in enumerate(sample):
            text = s[0]
            # last_paragraph = s[1]
            texts.append((text))
            input_ids_all = self.tok(text)["input_ids"][:100]
            assert len(input_ids_all) == 100, f"{len(input_ids_all)=}"
            # last_paragraph_ids = self.tok(last_paragraph)["input_ids"]
            # add the first token to the input_ids
            input_ids = torch.tensor([[input_ids_all[0]]]).to(device)
            input_ids_all = input_ids_all[1:]
            logits_all_tokens = []
            for i, token in enumerate(input_ids_all):
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True, use_cache=False)
                    torch.cuda.empty_cache()
                    gc.collect()
                    logits = outputs.logits
                    logits_all_tokens.append(logits[0, -1, :].clone().detach().cpu())
                    print(f"{logits[0, -1, :].clone().detach().cpu().shape=}")
                    # update the input_ids
                    if i < len(input_ids_all) - 1:
                        input_ids = torch.cat((input_ids, torch.tensor([[token]]).clone().detach().to(device)), dim=-1)

                    del outputs

                torch.cuda.empty_cache()
                gc.collect()


            # concat the last logit from each logits in the list
            logits_all_tokens = torch.stack(logits_all_tokens)
            assert logits_all_tokens.shape[0] == len(input_ids_all), f"{logits_all_tokens.shape=} {len(input_ids_all)=}"
            assert input_ids[0][1:].all() == torch.tensor(
                input_ids_all).all(), f"{input_ids[0][1:]=} {torch.tensor(input_ids_all)=}"
            # calculate the loss
            loss = calc_loss(input_ids, logits_all_tokens)
            list_loss.append(loss.item())
            pp = torch.exp(loss)
            if pp.item() == float("inf"):  # for not want to use inf
                pp = torch.tensor(0)
            wikipedia_pp_score.append(pp.item())
            if not self.use_mlp and not self.use_attention and not self.use_residual and not self.use_heads:
                with torch.no_grad():
                    output = self.model(input_ids, labels=input_ids.clone())
                    loss = output.loss
                direct_pp_score_no_intervention.append(torch.exp(loss).item())
            else:
                direct_pp_score_no_intervention.append(0)
            del input_ids
            torch.cuda.empty_cache()
            gc.collect()
            print(
                f"took {(time.time() - start_time) / (index + 1)} seconds to calculate wikipedia pp score on {index + 1} texts")
        # remove hooks
        self.model_wrap_remove_hooks(self.model, handles)
        print(f"{np.mean(wikipedia_pp_score)=} {np.std(wikipedia_pp_score)=}")
        print(f"{np.mean(direct_pp_score_no_intervention)=} {np.std(direct_pp_score_no_intervention)=}")
        print(f"{np.mean(list_loss)=} {np.std(list_loss)=}")
        print(f"took {time.time() - start_time} seconds to calculate wikipedia pp score")
        assert len(sample) == len(wikipedia_pp_score) == len(direct_pp_score_no_intervention) == len(texts)
        return wikipedia_pp_score, texts

    def add_good_shots(self, prompt: str, praphraze_prompt: str):
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
        # we will replace the first bad shot with the first good shot and so on
        bad_shots = [shot.split("\nwrong answer:")[0] for shot in praphraze_prompt.split("\nquestion:")][:-1]
        bad_shots_indexes = []
        for bad_shot in bad_shots:
            for i, shot in enumerate(self.list_bad_shot):
                if bad_shot in shot:
                    bad_shots_indexes.append(i)
                    break
        assert len(bad_shots_indexes) == 3, f"{len(bad_shots_indexes)=}"
        new_prompt = self.list_good_shot[bad_shots_indexes[0]] + self.list_good_shot[bad_shots_indexes[1]] + \
                     self.list_good_shot[bad_shots_indexes[2]] + prompt
        return new_prompt

