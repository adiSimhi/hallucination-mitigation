import gc
import os
import json
import pickle
import random
import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import ModelInside
from sklearn.svm import LinearSVC
# load f1_score
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(data_path):
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
    return data

def split_data_to_train_val_test(data_indexes,seed=None,static_dataset=None):
    """
    split data indexes to train val test
    :param data_indexes:
    :return:
    """
    random.seed(42)
    if seed is not None:
        random.seed(seed)
    random.shuffle(data_indexes)
    train = data_indexes[:int(0.7 * len(data_indexes))]
    val = data_indexes[int(0.7 * len(data_indexes)):]
    test = []
    assert len(train) + len(val) + len(test) == len(data_indexes)
    # print(f"{len(train)=} {len(val)=} {len(test)=}")
    return train, val, test

def split_data_to_train_val_test_for_all_data_types(data_split,seed=None,static_dataset=None):
    data_indeces = [i for i in range(len(data_split))]
    train_indexes, val_indexes, test_indexes = split_data_to_train_val_test(data_indeces,seed)
    train = [data_split[i] for i in train_indexes]
    val = [data_split[i] for i in val_indexes]
    test = [data_split[i] for i in test_indexes]
    return train, val, test



def linear_classifier(train_with, train_without, test_with=None, test_without=None,seed_train_val=None):
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
        if seed_train_val is not None:
            random_state = seed_train_val
        clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
        clf.fit(train_vectors_curr_layer, true_labels)
        classifier_for_layers.append(clf)
        if test_with is not None and test_without is not None:
            test_vectors_curr_layer = np.array([i[layer] for i in test])
            test_acc.append(clf.score(test_vectors_curr_layer, test_labels))
            test_labels_predicted.append(clf.predict(test_vectors_curr_layer))
    return classifier_for_layers, test_acc, test_labels_predicted, true_labels



def linear_classifier_3_options(train_1, train_2, train3 ,seed_train_val=None):
    """
    train a linear classifier on the data
    :param train_with: data with hallucinations
    :param train_without: data without hallucinations
    :return: classifier for each layer
    """
    # concatenate the data
    train = train_1 + train_2 + train3
    labels = [1] * len(train_1) + [0] * len(train_2) + [2] * len(train3)
    classifier_for_layers = []
    test_acc = []
    test_labels_predicted = []
    train, true_labels = shuffle(train, labels, random_state=0)
    for layer in range(len(train_1[0])):
        train_vectors_curr_layer = np.array([i[layer] for i in train])
        random_state = 0
        if seed_train_val is not None:
            random_state = seed_train_val
        clf = LinearSVC(random_state=random_state, tol=1e-5, dual=True, max_iter=1000000)
        # use neighbors classifier
        clf.fit(train_vectors_curr_layer, true_labels)
        classifier_for_layers.append(clf)
    return classifier_for_layers, test_acc, test_labels_predicted, true_labels


def classifier_dict_create_3_options(train_mlp_1, train_mlp_2, train_mlp_3, train_attention_1, train_attention_2,
                                     train_attention_3, train_residual_1, train_residual_2, train_residual_3,
                                     component_val, s=None):
    classifiers_dict = {}
    if train_mlp_1 is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier_3_options(train_mlp_1, train_mlp_2,
                                                                                              train_mlp_3,
                                                                                              seed_train_val=s)
        classifiers_dict["mlp"] = classifiers
    if train_attention_1 is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier_3_options(train_attention_1,
                                                                                      train_attention_2,
                                                                                      train_attention_3,
                                                                                      seed_train_val=s)
        classifiers_dict["attention"] = classifiers
    if train_residual_1 is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier_3_options(train_residual_1,
                                                                                      train_residual_2,
                                                                                      train_residual_3,
                                                                                      seed_train_val=s)
        classifiers_dict["residual"] = classifiers
    one_dict = {key: [] for key in classifiers_dict.keys()}
    f1_dict = {key: [] for key in classifiers_dict.keys()}
    for key in classifiers_dict.keys():
        if key != "heads":
            test_1 = component_val[key][0]
            test_2 = component_val[key][1]
            test_3 = component_val[key][2]

            test = test_1 + test_2 + test_3
            test_labels = [1] * len(test_1) + [0] * len(test_2) + [2] * len(test_3)
            test, test_labels = shuffle(test, test_labels, random_state=0)
            for layer in range(len(classifiers_dict[key])):
                classifier = classifiers_dict[key][layer]
                test_vectors_curr_layer = np.array([i[layer] for i in test])
                test_predictions = classifier.predict(test_vectors_curr_layer)
                acc = accuracy_score(test_labels, test_predictions)
                one_dict[key].append(acc)
                # print the percentage per each two of the three classes of the number of examples that in the test set is labeled as one and in the prediction is labeled as the other
                print(f"{key=} {layer=}")
                for i in range(3):
                    for i2 in range(3):
                        # print(f"{test_labels=} {test_predictions=}")
                        if i < i2:
                            # print(f"{np.sum(np.logical_or(np.array(test_labels) == i, np.array(test_labels) == i2))=}")
                            print(f"{i=} {i2=} {1-((np.sum(np.logical_and(np.array(test_labels) == i, test_predictions == i2)) +np.sum(np.logical_and(np.array(test_labels) == i2, test_predictions == i))) / np.sum(np.logical_or(np.array(test_labels) == i, np.array(test_labels) == i2)))}")

                f1 = f1_score(test_labels, test_predictions, average='weighted')
                f1_dict[key].append(f1)
    return one_dict, f1_dict


def classifier_dict_create(train_mlp_with, train_mlp_without, train_attention_with, train_attention_without,
                            train_residual_with, train_residual_without,component_val, s=None):
    classifiers_dict = {}
    if train_mlp_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_mlp_with, train_mlp_without,
                                                                                      seed_train_val=s)
        classifiers_dict["mlp"] = classifiers
    if train_attention_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_attention_with,
                                                                                      train_attention_without,
                                                                                      seed_train_val=s)
        classifiers_dict["attention"] = classifiers
    if train_residual_with is not None:
        classifiers, test_acc, test_labels_predicted, true_labels = linear_classifier(train_residual_with,
                                                                                      train_residual_without,
                                                                                      seed_train_val=s)
        classifiers_dict["residual"] = classifiers
    one_dict = {key: [] for key in classifiers_dict.keys()}
    f1_dict = {key: [] for key in classifiers_dict.keys()}
    for key in classifiers_dict.keys():
        if key != "heads":
            test_with = component_val[key][0]
            test_without = component_val[key][1]
            test = test_with + test_without
            test_labels = [1] * len(test_with) + [0] * len(test_without)
            test, test_labels = shuffle(test, test_labels, random_state=0)
            for layer in range(len(classifiers_dict[key])):
                classifier = classifiers_dict[key][layer]
                test_vectors_curr_layer = np.array([i[layer] for i in test])
                test_predictions = classifier.predict(test_vectors_curr_layer)
                acc = accuracy_score(test_labels, test_predictions)
                one_dict[key].append(acc)
                f1 = f1_score(test_labels, test_predictions, average='weighted')
                f1_dict[key].append(f1)
    return one_dict, f1_dict

def indices_not_use(data_static, data_with_hall, data_without_hall,train_indexes, val_indexes,s):
    data_static_train_prompt = [(data_static[i][0],i) for i in train_indexes]
    data_with_val_prompt = [data_with_hall[i][0] for i in val_indexes]
    data_without_val_prompt = [data_without_hall[i][0] for i in val_indexes]
    assert len(data_static_train_prompt) == len(train_indexes) , f"{len(data_static_train_prompt)} != {len(train_indexes)}"
    assert len(data_with_val_prompt) == len(val_indexes) and len(data_without_val_prompt) == len(val_indexes), f"{len(data_with_val_prompt)} != {len(val_indexes)} {len(data_without_val_prompt)} != {len(val_indexes)}"
    non_use_static_indeces = [data_static_train_prompt[i][1] for i in range(len(data_static_train_prompt)) if data_static_train_prompt[i][0] in data_with_val_prompt or data_static_train_prompt[i][0] in data_without_val_prompt]
    # print(f"{len(non_use_static_indeces)=}")
    train_static_indexes = [i for i in train_indexes if i not in non_use_static_indeces]
    # print(f"{len(train_static_indexes)=}")
    # add len(non_use_static_indeces) from the val indexes to the new_static_train to make it the same size as the train indexes, make sure that the new indexes are not in in data_with_val_prompt or data_without_val_prompt
    for i in range(len(non_use_static_indeces)):
        new_index = random.sample(val_indexes, 1)[0]
        example_static = data_static[new_index][0]
        while example_static in data_with_val_prompt or example_static in data_without_val_prompt or new_index in train_static_indexes:

            new_index = random.sample(val_indexes, 1)[0]
            example_static = data_static[new_index][0]
        train_static_indexes.append(new_index)
    assert len(train_static_indexes) == len(train_indexes) == len(set(train_static_indexes)), f"{len(train_static_indexes)} != {len(train_indexes)} {len(set(train_static_indexes))}"
    for i in train_static_indexes:
        assert data_static[i][0] not in data_with_val_prompt and data_static[i][0] not in data_without_val_prompt, f"{data_static[i][0]=} {data_with_val_prompt=} {data_without_val_prompt=}"
    return train_indexes, val_indexes, []






def plot_classification_graphs_on_pre_answer(threshold, model_name, dataset_size=500, dataset_name="disentQA", alpha=5,
                                      concat_answer=False, seed=None, static_dataset=False, concat_answer_test=True, static_dataset_test=False,alice=False,alice_test=False):
    final_classification_dict_static = []
    final_classification_dict = []
    seeds = [None,100,200]

    data_static = load_dataset(f"datasets/Static{dataset_name[0].upper() + dataset_name[1:]}.json")
    data_with_hall = load_dataset(f"datasets/Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    data_without_hall = load_dataset(f"datasets/NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    if alice_test:
        data_static = load_dataset(f"datasets/AliceStatic{dataset_name[0].upper() + dataset_name[1:]}.json")
        data_with_hall = load_dataset(
            f"datasets/AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
        data_without_hall = load_dataset(
            f"datasets/AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
    for s in seeds:
        MLPCheck = ModelInside.ModelInside("results/",
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=1.0, concat_answer=concat_answer_test,static_dataset=static_dataset_test,alice=alice_test)
        all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.load_all_data()
        example_indeces = [i for i in range(len(all_mlp_vector_with_hall))]
        train, val, test = split_data_to_train_val_test(example_indeces, s)
        train_mlp_with , val_mlp_with,_ = split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall / np.linalg.norm(all_mlp_vector_with_hall, axis=2)[:, :, np.newaxis],s)
        train_mlp_without , val_mlp_without,_ = [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train], [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in val], [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in test]
        train_attention_with , val_attention_with,_ = split_data_to_train_val_test_for_all_data_types(all_attention_vector_with_all / np.linalg.norm(all_attention_vector_with_all, axis=2)[:, :, np.newaxis],s)
        train_attention_without , val_attention_without,_ = [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train], [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in val], [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in test]
        train_residual_with , val_residual_with,_ = split_data_to_train_val_test_for_all_data_types(all_residual_with/ np.linalg.norm(all_residual_with, axis=2)[:, :, np.newaxis],s)
        train_residual_without , val_residual_without,_ = [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in train], [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in val], [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in test]
        component_val = {"mlp": [val_mlp_with, val_mlp_without], "attention": [val_attention_with, val_attention_without], "residual": [val_residual_with, val_residual_without]}
        acc,f1 = classifier_dict_create(train_mlp_with, train_mlp_without, train_attention_with, train_attention_without, train_residual_with, train_residual_without, component_val, s)
        final_classification_dict.append(acc)
        assert len(train_mlp_with)==len(train_mlp_without)==len(train_attention_with)==len(train_attention_without)==len(train_residual_with)==len(train_residual_without), f"{len(train_mlp_with)} != {len(train_mlp_without)} {len(train_attention_with)} != {len(train_attention_without)} {len(train_residual_with)} != {len(train_residual_without)}"
        assert len(val_mlp_with)==len(val_mlp_without)==len(val_attention_with)==len(val_attention_without)==len(val_residual_with)==len(val_residual_without), f"{len(val_mlp_with)} != {len(val_mlp_without)} {len(val_attention_with)} != {len(val_attention_without)} {len(val_residual_with)} != {len(val_residual_without)}"
        del MLPCheck
        gc.collect()
        torch.cuda.empty_cache()
        MLPCheck = ModelInside.ModelInside("results/",
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=1.0, concat_answer=concat_answer, static_dataset=static_dataset,alice=alice)
        train_indexes, val_indexes, test_indexes = split_data_to_train_val_test([i for i in range(len(all_mlp_vector_with_hall))], s)
        assert test_indexes == test, f"{test_indexes=} {test=}"
        assert val_indexes == val, f"{val_indexes=} {val=}"
        assert train_indexes == train, f"{train_indexes=} {train=}"
        print(f"{len(train_indexes)=} {len(val_indexes)=} {len(test_indexes)=}")
        all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.load_all_data()
        if static_dataset:
            print(f"training on static dataset")
            assert len(all_mlp_vector_with_hall)==1000, f"{len(all_mlp_vector_with_hall)} != 1000"
            train_static_indexes, val_static_indexes, test_static_indexes = indices_not_use(data_static, data_with_hall, data_without_hall, train_indexes, val_indexes,s)

        elif (alice and not alice_test) or (not alice and alice_test):
            # make sure it does not train on prompts that are in the validation set
            random.seed(s)
            train_static_indexes = random.sample([i for i in range(len(all_mlp_vector_with_hall))],
                                                 min(len(train_mlp_with), len(all_mlp_vector_with_hall)))
            data_with_hall_alice = load_dataset(
                f"datasets/AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
            data_without_hall_alice = load_dataset(
                f"datasets/AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
            if alice_test:
                data_with_hall_alice = load_dataset(
                    f"datasets/Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
                data_without_hall_alice = load_dataset(
                    f"datasets/NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json")
            print(f"{data_with_hall[0]=}, print({val_indexes=}")
            data_with_val_prompt = [data_with_hall[i][0] for i in val_indexes]
            data_without_val_prompt = [data_without_hall[i][0] for i in val_indexes]
            non_use_static_indeces = [i for i in train_static_indexes if
                                      data_with_hall_alice[i][0] in data_with_val_prompt or data_with_hall_alice[i][
                                          0] in data_without_val_prompt or data_without_hall_alice[i][
                                          0] in data_with_val_prompt or data_without_hall_alice[i][
                                          0] in data_without_val_prompt]
            train_static_indexes = [i for i in train_static_indexes if i not in non_use_static_indeces]
            for i in train_static_indexes:
                assert data_with_hall_alice[i][0] not in data_with_val_prompt and data_with_hall_alice[i][0] not in data_without_val_prompt and data_without_hall_alice[i][0] not in data_with_val_prompt and data_without_hall_alice[i][0] not in data_without_val_prompt, f"{data_with_hall_alice[i][0]=} "
            print(f"alice: {len(train_static_indexes)=} {len(non_use_static_indeces)=} {len(train_mlp_with)=}")
            for i in range(len(non_use_static_indeces)):
                sampling_index = [i for i in range(len(all_mlp_vector_with_hall)) if (i not in train_static_indexes and i not in non_use_static_indeces)]
                if len(sampling_index) == 0:
                    # print(f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                    break
                new_index = random.sample(sampling_index, 1)[0]
                while data_with_hall_alice[new_index][0] in data_with_val_prompt or data_with_hall_alice[new_index][
                    0] in data_without_val_prompt or data_without_hall_alice[new_index][0] in data_with_val_prompt or \
                        data_without_hall_alice[new_index][0] in data_without_val_prompt:
                    non_use_static_indeces.append(new_index)
                    sampling_index = [i for i in range(len(all_mlp_vector_with_hall)) if (i not in train_static_indexes and i not in non_use_static_indeces)]
                    if len(sampling_index) == 0:
                        # print(f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                        new_index = None
                        break
                    new_index = random.sample(sampling_index, 1)[0]
                    if len(non_use_static_indeces) + len(train_static_indexes) == len(all_mlp_vector_with_hall):
                        # print(
                        #     f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                        new_index = None
                        break
                if new_index is not None:
                    train_static_indexes.append(new_index)
                if len(non_use_static_indeces) + len(train_static_indexes) == len(all_mlp_vector_with_hall):
                    # print(
                    #     f" pass {len(non_use_static_indeces)=} {len(train_static_indexes)=} {len(all_mlp_vector_with_hall)=}")
                    break
            for i in train_static_indexes:
                assert data_with_hall_alice[i][0] not in data_with_val_prompt and data_with_hall_alice[i][0] not in data_without_val_prompt and data_without_hall_alice[i][0] not in data_with_val_prompt and data_without_hall_alice[i][0] not in data_without_val_prompt, f"{data_with_hall_alice[i][0]=} "
            assert len(train_static_indexes) == len(set(train_static_indexes)), f"{len(train_static_indexes)} != {len(train_mlp_with)} {len(set(train_static_indexes))}"
            print(f"alice: {len(train_static_indexes)=}")
        else:
            train_static_indexes = train_indexes



        train_mlp_with = [all_mlp_vector_with_hall[i]/np.linalg.norm(all_mlp_vector_with_hall[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        train_mlp_without = [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        train_attention_with = [all_attention_vector_with_all[i]/np.linalg.norm(all_attention_vector_with_all[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        train_attention_without = [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        train_residual_with = [all_residual_with[i]/np.linalg.norm(all_residual_with[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        train_residual_without = [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in train_static_indexes]
        assert len(train_mlp_with)==len(train_mlp_without)==len(train_attention_with)==len(train_attention_without)==len(train_residual_with)==len(train_residual_without), f"{len(train_mlp_with)} != {len(train_mlp_without)} {len(train_attention_with)} != {len(train_attention_without)} {len(train_residual_with)} != {len(train_residual_without)}"
        # print(f"{len(train_mlp_with)=} {len(train_mlp_without)=} {len(train_attention_with)=} {len(train_attention_without)=} {len(train_residual_with)=} {len(train_residual_without)=}")
        # print(f"{np.shape(train_mlp_with[0])=}")

        one_dict,f1_Score = classifier_dict_create(train_mlp_with, train_mlp_without, train_attention_with, train_attention_without,
                                          train_residual_with, train_residual_without, component_val, s)
        final_classification_dict_static.append(one_dict)
    print(f"{final_classification_dict=}")
    final_dict = {"residual": [], "mlp": [], "attention": []}
    final_dict_std = {"residual": [], "mlp": [], "attention": []}
    final_dict_static = {"residual_static": [], "mlp_static": [], "attention_static": []}
    final_dict_std_static = {"residual_static": [], "mlp_static": [], "attention_static": []}

    for key in final_classification_dict[0].keys():

        for i in range(len(final_classification_dict[0][key])):
            final_dict[key].append(np.mean([c[key][i] * 100 for c in final_classification_dict]))
            final_dict_std[key].append(np.std([c[key][i] * 100 for c in final_classification_dict]))
            final_dict_static[key + "_static"].append(np.mean([c[key][i] * 100 for c in final_classification_dict_static]))
            final_dict_std_static[key + "_static"].append(np.std([c[key][i] * 100 for c in final_classification_dict_static]))


    return final_dict, final_dict_std, final_dict_static, final_dict_std_static






def plot_all_models_graphs_together(dict_final_dict, dict_final_std,dataset_name,title,y_lim=50,static_acc_dict=None,static_acc_std_dict=None,post_non_static_on_pre=None,post_non_static_on_pre_std=None,alice=False, bad_shot_vs_alice=False):


    for i, key in enumerate(dict_final_dict[list(dict_final_dict.keys())[0]].keys()):
        plt.figure()
        plt.xlabel("layer", fontsize=15)
        plt.ylabel("accuracy", fontsize=15)
        plt.ylim(20, 100)
        plt.yticks(np.arange(20, 100, 10))
        #choose three colors that are friendly to colorblind people
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        # add shapes
        # shapes = ['o', 's', 'D']
        line_explanation = {}
        for j, model_name in enumerate(dict_final_dict.keys()):
            label_name = model_name.replace("mistralai/Mistral-7B-v0.3", "Mistral").replace("meta-llama/Meta-Llama-3.1-8B","Llama").replace("google/gemma-2-9b","Gemma")
            print(f"{dict_final_dict[model_name][key]=}")
            current_label = label_name
            if bad_shot_vs_alice:
                line_explanation["solid"] = "Alice-Bob"
            elif static_acc_dict is not None and post_non_static_on_pre is None:
                line_explanation["solid"] = "specific"
            elif post_non_static_on_pre is not None:
                line_explanation["solid"] = "pre-specific"
            plt.plot(dict_final_dict[model_name][key], label=current_label, color=colors[j], linestyle="solid")
            # add the std
            plt.fill_between(range(len(dict_final_dict[model_name][key])), np.array(dict_final_dict[model_name][key]) - np.array(dict_final_std[model_name][key]),
                             np.array(dict_final_dict[model_name][key]) + np.array(dict_final_std[model_name][key]), alpha=0.3, color=colors[j])
            if static_acc_dict is not None and bad_shot_vs_alice==False and post_non_static_on_pre is None:
                # print(f"{static_acc_dict[model_name][key+'_static']=}")
                line_explanation["dashed"] = "generic"
                plt.plot(static_acc_dict[model_name][key+'_static'], color=colors[j], linestyle="dashed")
                # add the std
                plt.fill_between(range(len(static_acc_dict[model_name][key+'_static'])), np.array(static_acc_dict[model_name][key+'_static']) - np.array(static_acc_std_dict[model_name][key+'_static']),
                                np.array(static_acc_dict[model_name][key+'_static']) + np.array(static_acc_std_dict[model_name][key+'_static']), alpha=0.3, color=colors[j])
            if  static_acc_dict is not None and bad_shot_vs_alice  is True:
                # print(f"{static_acc_dict[model_name][key + '_static']=}")
                line_explanation["dashed"] = "bad_shots"
                plt.plot(static_acc_dict[model_name][key + '_static'], color=colors[j],
                         linestyle="dashed")
                # add the std
                plt.fill_between(range(len(static_acc_dict[model_name][key + '_static'])),
                                 np.array(static_acc_dict[model_name][key + '_static']) - np.array(
                                     static_acc_std_dict[model_name][key + '_static']),
                                 np.array(static_acc_dict[model_name][key + '_static']) + np.array(
                                     static_acc_std_dict[model_name][key + '_static']), alpha=0.3, color=colors[j])

            if static_acc_dict is not None and bad_shot_vs_alice==False and post_non_static_on_pre is not None:
                # print(f"{static_acc_dict[model_name][key+'_static']=}")
                line_explanation["dashed"] = "post-generic"
                plt.plot(static_acc_dict[model_name][key+'_static'],color=colors[j], linestyle="dashed")
                # add the std
                plt.fill_between(range(len(static_acc_dict[model_name][key+'_static'])), np.array(static_acc_dict[model_name][key+'_static']) - np.array(static_acc_std_dict[model_name][key+'_static']),
                                np.array(static_acc_dict[model_name][key+'_static']) + np.array(static_acc_std_dict[model_name][key+'_static']), alpha=0.3, color=colors[j])

            if post_non_static_on_pre is not None:
                # print(f"{post_non_static_on_pre[model_name][key+'_static']=}")
                line_explanation["dotted"] = "post-specific"
                plt.plot(post_non_static_on_pre[model_name][key+'_static'], color=colors[j], linestyle="dotted")
                # add the std
                plt.fill_between(range(len(post_non_static_on_pre[model_name][key+'_static'])), np.array(post_non_static_on_pre[model_name][key+'_static']) - np.array(post_non_static_on_pre_std[model_name][key+'_static']),
                                np.array(post_non_static_on_pre[model_name][key+'_static']) + np.array(post_non_static_on_pre_std[model_name][key+'_static']), alpha=0.3, color=colors[j])
        if post_non_static_on_pre is not None:
            plt.text(0, y_lim, 'Baseline', color='black', va='center', ha='center', backgroundcolor='white', fontsize=10)
        else:
            plt.axhline(y=y_lim, color='black', linestyle='--')

        # Place model legend above the line style legend
        model_legend = plt.legend(loc='lower left', ncol=1,
                                  fontsize=15)
        plt.gca().add_artist(model_legend)

        # add the labels of the line_explanation to the plot at a line at the bottom right with the type of line
        # e.g solid, dashed, dotted and the explanation
        # plot them in one line at the bottom right
        line_style_handles = []
        if len(line_explanation.keys()) > 1:
            for k, value in line_explanation.items():
                line_style_handles.append(plt.plot([], [], color='black', linestyle=k, label=value)[0])


            # Place line style legend at the bottom
            line_style_legend = plt.legend(handles=line_style_handles, loc='lower left', bbox_to_anchor=(0.33, 0),
                                           ncol=1, fontsize=15)
            plt.gca().add_artist(line_style_legend)



        # plt.legend(fontsize=15)
        plt.grid()
        path_to_save = f"results/all_models_3_options_{key}_{dataset_name}_{title}_{alice}.pdf"

        plt.savefig(path_to_save)
        plt.close()


def plot_classification_graphs_type1_vs_type3_know(threshold, model_name, dataset_size=500, dataset_name="disentQA", alpha=5,
                                      concat_answer=False, seed=None, static_dataset=False, concat_answer_test=True, static_dataset_test=False, alice=False):

    path_else = f"datasets/General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json"
    if alice:
        path_else = f"datasets/AliceGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}.json"
    final_classification_dict = []
    seeds = [None,100,200]
    path = f"results/{model_name.replace('/', '_')}" + f'{"/"}' + (
        f"{dataset_name}/{threshold}/concat_answer{concat_answer}_size{dataset_size}/"
        f"classifier_dict_False_False_False_False.pkl")
    for s in seeds:
        MLPCheck = ModelInside.ModelInside("results/",
                                           None,
                                           None,
                                           model_name=model_name, dataset_size=dataset_size,
                                           dataset_name=dataset_name,
                                           threshold_of_data=1.0, concat_answer=concat_answer_test,static_dataset=static_dataset_test,alice=alice)
        all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.load_all_data()
        example_indeces = [i for i in range(len(all_mlp_vector_with_hall))]
        train,val,test = split_data_to_train_val_test(example_indeces,s)
        train_mlp_with , val_mlp_with,_ = split_data_to_train_val_test_for_all_data_types(all_mlp_vector_with_hall/ np.linalg.norm(all_mlp_vector_with_hall, axis=2)[:, :, np.newaxis],s)
        train_attention_with , val_attention_with,_ = split_data_to_train_val_test_for_all_data_types(all_attention_vector_with_all/ np.linalg.norm(all_attention_vector_with_all, axis=2)[:, :, np.newaxis],s)
        train_residual_with , val_residual_with,_ = split_data_to_train_val_test_for_all_data_types(all_residual_with/ np.linalg.norm(all_residual_with, axis=2)[:, :, np.newaxis],s)
        train_mlp_without , val_mlp_without = [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train], [all_mlp_vector_without_hall[i]/np.linalg.norm(all_mlp_vector_without_hall[i], axis=1)[:, np.newaxis] for i in val]
        train_attention_without , val_attention_without = [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in train], [all_attention_vector_without_hall[i]/np.linalg.norm(all_attention_vector_without_hall[i], axis=1)[:, np.newaxis] for i in val]
        train_residual_without , val_residual_without = [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in train], [all_residual_without[i]/np.linalg.norm(all_residual_without[i], axis=1)[:, np.newaxis] for i in val]

        all_mlp_vector_type1, all_attention_vector_type1, heads_vectors_type1, all_residual_vectors_type1 = MLPCheck.get_type_1_data(path_else)

        train_mlp_type1 , val_mlp_type1 = [all_mlp_vector_type1[i]/np.linalg.norm(all_mlp_vector_type1[i], axis=1)[:, np.newaxis] for i in train], [all_mlp_vector_type1[i]/np.linalg.norm(all_mlp_vector_type1[i], axis=1)[:, np.newaxis] for i in val]
        train_attention_type1 , val_attention_type1 = [all_attention_vector_type1[i]/np.linalg.norm(all_attention_vector_type1[i], axis=1)[:, np.newaxis] for i in train], [all_attention_vector_type1[i]/np.linalg.norm(all_attention_vector_type1[i], axis=1)[:, np.newaxis] for i in val]
        train_residual_type1 , val_residual_type1 = [all_residual_vectors_type1[i]/np.linalg.norm(all_residual_vectors_type1[i], axis=1)[:, np.newaxis] for i in train], [all_residual_vectors_type1[i]/np.linalg.norm(all_residual_vectors_type1[i], axis=1)[:, np.newaxis] for i in val]

        assert len(train_mlp_with)==len(train_mlp_without)==len(train_attention_with)==len(train_attention_without)==len(train_residual_with)==len(train_residual_without) == len(train_mlp_type1) == len(train_attention_type1) == len(train_residual_type1), f"{len(train_mlp_with)} != {len(train_mlp_without)} {len(train_attention_with)} != {len(train_attention_without)} {len(train_residual_with)} != {len(train_residual_without)} {len(train_mlp_type1)} != {len(train_attention_type1)} {len(train_residual_type1)}"
        assert len(val_mlp_with)==len(val_mlp_without)==len(val_attention_with)==len(val_attention_without)==len(val_residual_with)==len(val_residual_without) == len(val_mlp_type1) == len(val_attention_type1) == len(val_residual_type1), f"{len(val_mlp_with)} != {len(val_mlp_without)} {len(val_attention_with)} != {len(val_attention_without)} {len(val_residual_with)} != {len(val_residual_without)} {len(val_mlp_type1)} != {len(val_attention_type1)} {len(val_residual_type1)}"


        component_val = {"mlp": [val_mlp_with, val_mlp_without, val_mlp_type1], "attention": [val_attention_with, val_attention_without, val_attention_type1], "residual": [val_residual_with, val_residual_without, val_residual_type1]}
        # component_val = {"residual": [val_residual_with, val_residual_without, val_residual_type1]}
        # acc,f1score = classifier_dict_create_3_options(None, None,None, None, None, None, train_residual_with, train_residual_without, train_residual_type1, component_val, s)
        acc, f1score = classifier_dict_create_3_options(train_mlp_1=train_mlp_with, train_mlp_2=train_mlp_without, train_mlp_3=train_mlp_type1, train_attention_1=train_attention_with, train_attention_2=train_attention_without, train_attention_3=train_attention_type1, train_residual_1=train_residual_with, train_residual_2=train_residual_without, train_residual_3=train_residual_type1, component_val=component_val, s = s)
        final_classification_dict.append(acc)

    print(f"{final_classification_dict=}")
    final_dict = {"residual": [], "mlp": [], "attention": []}
    final_dict_std = {"residual": [], "mlp": [], "attention": []}
    final_f1 = {"residual": [], "mlp": [], "attention": []}
    final_f1_std = {"residual": [], "mlp": [], "attention": []}

    for key in final_classification_dict[0].keys():

        for i in range(len(final_classification_dict[0][key])):
            final_dict[key].append(np.mean([c[key][i] * 100 for c in final_classification_dict]))
            final_dict_std[key].append(np.std([c[key][i] * 100 for c in final_classification_dict]))
    # print(f"{classifiers=}")
    # create an average of the different seeds and an std
    print(f"{final_dict=}")
    print(f"{final_dict_std=}")
    return final_dict, final_dict_std




# if __name__ == "__main__":

def run_plot_results(args_post_answer=False, args_pre_answer=False, args_alice=False, args_type1_vs_type3_know=False, args_alice_vs_bad_shot=False):



    dataset_name = ["natural_qa_no_context", "trivia_qa_no_context"]
    models = ["mistralai/Mistral-7B-v0.3","meta-llama/Meta-Llama-3.1-8B","google/gemma-2-9b"]
    for name in dataset_name:
        models_acc_dict = {}
        models_acc_std_dict = {}
        models_f1_dict = {}
        models_f1_dict_std = {}
        static_acc_dict = {}
        static_acc_std_dict = {}
        post_non_static_on_pre = {}
        post_non_static_on_pre_std = {}
        final_f1 = None
        final_dict_post = None
        final_dict_static = None
        bad_shot_vs_alice = False
        alice = False
        for model in models:

            print(f"{model=} dataset={name}",flush=True)
            print(f"{args_post_answer=} {args_pre_answer=} {args_alice=} {args_type1_vs_type3_know=} {args_alice_vs_bad_shot=}")


            # post answer
            if args_post_answer and not args_alice:
                alice = False
                final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_pre_answer(1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                                                         concat_answer=True,static_dataset=True,concat_answer_test=True, static_dataset_test=False, seed=None, alice=alice, alice_test=alice)

            #post answer alice
            elif args_pre_answer and args_alice:
                alice = True
                final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_pre_answer(1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                                                         concat_answer=True,static_dataset=True,concat_answer_test=True, static_dataset_test=False, seed=None, alice=alice, alice_test=alice)



            #pre answer
            elif args_pre_answer and not args_alice:
                final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_pre_answer(1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                                                         concat_answer=True, seed=None, static_dataset=True, static_dataset_test=False, concat_answer_test=False)
                final_dict, final_dict_std, final_dict_post, final_dict_std_post = plot_classification_graphs_on_pre_answer(1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                                                         concat_answer=True, seed=None, static_dataset=False,
                                                         static_dataset_test=False, concat_answer_test=False)

            # pre answer alice
            elif args_pre_answer and args_alice:
                alice = True
                final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_pre_answer(
                    1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                    concat_answer=True, seed=None, static_dataset=True, static_dataset_test=False, concat_answer_test=False,alice=alice,alice_test=alice)
                final_dict, final_dict_std, final_dict_post, final_dict_std_post = plot_classification_graphs_on_pre_answer(
                    1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                    concat_answer=True, seed=None, static_dataset=False,
                    static_dataset_test=False, concat_answer_test=False,alice=alice,alice_test=alice)

            #alice pre answer tested on bad-shot pre
            elif args_alice_vs_bad_shot:
                alice = True
                bad_shot_vs_alice = True
                print(f"{alice=} {bad_shot_vs_alice=}")
                final_dict, final_dict_std, final_dict_static, final_dict_std_static = plot_classification_graphs_on_pre_answer(
                    1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                    concat_answer=True, seed=None, static_dataset=False, static_dataset_test=False, concat_answer_test=True,alice_test=alice)


            # type1 vs type3 vs known
            elif args_type1_vs_type3_know:
                alice = False
                if args_alice:
                    alice = True
                final_dict, final_dict_std = plot_classification_graphs_type1_vs_type3_know(1.0, model, dataset_size=1000, dataset_name=name, alpha=5,
                                                          concat_answer=False, seed=None, static_dataset=False,
                                                          static_dataset_test=False, concat_answer_test=True, alice=alice
                                                          )
            models_acc_dict[model] = final_dict
            models_acc_std_dict[model] = final_dict_std
            if final_dict_static:
                static_acc_dict[model] = final_dict_static
                static_acc_std_dict[model] = final_dict_std_static
            if final_dict_post:
                post_non_static_on_pre[model] = final_dict_post
                post_non_static_on_pre_std[model] = final_dict_std_post
        print(f"{models_acc_dict=}")

        if static_acc_dict!={} and alice is True and bad_shot_vs_alice: #if we have static results
            plot_all_models_graphs_together(models_acc_dict, models_acc_std_dict,name,title="alice_vs_bad_shot_tested_post_answer_on_alice",y_lim=50,static_acc_dict=static_acc_dict,static_acc_std_dict=static_acc_std_dict,alice=alice,bad_shot_vs_alice=bad_shot_vs_alice)


        elif static_acc_dict!={} and post_non_static_on_pre!={}: #if we have static results
            plot_all_models_graphs_together(models_acc_dict, models_acc_std_dict,name,title="static_vs_non_static_non_static_post_tested_pre_answer",y_lim=50,static_acc_dict=static_acc_dict,static_acc_std_dict=static_acc_std_dict,post_non_static_on_pre=post_non_static_on_pre,post_non_static_on_pre_std=post_non_static_on_pre_std,alice=alice,bad_shot_vs_alice=bad_shot_vs_alice)
        elif static_acc_dict!={}:
            plot_all_models_graphs_together(models_acc_dict, models_acc_std_dict, name,
                                            title="static_vs_non_static_tested_post_answer", y_lim=50,
                                            static_acc_dict=static_acc_dict, static_acc_std_dict=static_acc_std_dict,alice=alice,bad_shot_vs_alice=bad_shot_vs_alice)
        else:
            plot_all_models_graphs_together(models_acc_dict, models_acc_std_dict, name,
                                                title="type1_vs_type3_vs_know", y_lim=33,alice=alice,bad_shot_vs_alice=bad_shot_vs_alice)


