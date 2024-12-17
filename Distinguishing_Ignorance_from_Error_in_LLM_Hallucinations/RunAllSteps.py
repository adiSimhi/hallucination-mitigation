import argparse
import datetime
import gc
import json
import os
import subprocess
import torch
from ModelInside import ModelInside
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

initial_dataset_path = "datasets/"

ending = ".json"


def create_dataset(dataset_name, dataset_path=None, threshold=1.0, model_name="GOAT-AI/GOAT-7B-Community", alice=False):
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

    from DatasetCreationWithoutContext import CreateDataset

    # create static dataset
    static_path = f"{initial_dataset_path}Static{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    test_set = f"{initial_dataset_path}TestStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_path):
        print(f"{static_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name="mistralai/Mistral-7B-v0.3",
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, static_dataset=True)
        #split dataset_creation.static_final_dataset into train and test where the test needs to have 200 examples
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_path)
        # empty the memory
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()
    static_alice_path = f"{initial_dataset_path}AliceStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if not os.path.exists(static_alice_path):
        print(f"{static_alice_path} does not exist")
        dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                         hall_save_path=None,
                                         non_hall_save_path=None,
                                         general_save_path=None, dataset_name=dataset_name,
                                         static_path=static_path, alice_story=True, static_dataset=True)
        dataset_creation.save_data(data=dataset_creation.static_final_dataset, path=static_alice_path)
        del dataset_creation
        torch.cuda.empty_cache()
        gc.collect()

    # create specific dataset
    hall_save_path = f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    non_hall_save_path = f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    general_save_path = f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    if alice:
        hall_save_path = f"{initial_dataset_path}AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        non_hall_save_path = f"{initial_dataset_path}AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        general_save_path = f"{initial_dataset_path}AliceGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    dataset_creation = CreateDataset(dataset_path, threshold, model_name=model_name,
                                     hall_save_path=hall_save_path,
                                     non_hall_save_path=non_hall_save_path,
                                     general_save_path=general_save_path,dataset_name=dataset_name,
                                     static_path=static_path,alice_story=alice)
    # create dataset
    dataset_creation.save_data(data = dataset_creation.non_hall_dataset, path = non_hall_save_path)
    dataset_creation.save_data(data = dataset_creation.hall_dataset, path = hall_save_path)
    dataset_creation.save_data(data = dataset_creation.general_dataset, path = general_save_path)


def run_initial_test_on_dataset(threshold, model_name="GOAT-AI/GOAT-7B-Community", dataset_size=1000,
                                dataset_name="disentQA", concat_answer=False, alice=False):
    print(
        f"threshold {threshold} model {model_name} dataset size {dataset_size} dataset name {dataset_name} concat_answer {concat_answer}")
    print(f"{initial_dataset_path=}")
    path_with = f"{initial_dataset_path}Hallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_without = f"{initial_dataset_path}NonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_general = f"{initial_dataset_path}General{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
    path_static = f"{initial_dataset_path}Static{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    if alice:
        path_with = f"{initial_dataset_path}AliceHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_without = f"{initial_dataset_path}AliceNonHallucinate{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_general = f"{initial_dataset_path}AliceGeneral{dataset_name[0].upper() + dataset_name[1:]}WithThreshold{threshold}_{model_name.replace('/', '_')}{ending}"
        path_static = f"{initial_dataset_path}AliceStatic{dataset_name[0].upper() + dataset_name[1:]}{ending}"
    MLPCheck = ModelInside("results/",
                           data_path_without_hallucinations=path_without,
                            data_path_with_hallucinations=path_with,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=False, static_dataset=False,alice=alice)
    all_mlp_vector_with_hall, all_attention_vector_with_all, all_mlp_vector_without_hall, all_attention_vector_without_hall, heads_vectors_with, heads_vectors_without, all_residual_with, all_residual_without = MLPCheck.generate_data()

    del MLPCheck
    torch.cuda.empty_cache()
    gc.collect()
    MLPCheck = ModelInside("results/",
                           path_static,
                           path_static,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=True,static_dataset=True, alice=alice)
    all_mlp_vector_with_static, all_attention_vector_with_static, all_mlp_vector_without_static, all_attention_vector_without_static, heads_vectors_with_static, heads_vectors_without_static, all_residual_with_static, all_residual_without_static = MLPCheck.generate_data()
    del MLPCheck
    torch.cuda.empty_cache()
    gc.collect()
    MLPCheck = ModelInside("results/",
                            data_path_without_hallucinations=path_without,
                            data_path_with_hallucinations=path_with,
                           model_name=model_name, dataset_size=dataset_size, dataset_name=dataset_name,
                           threshold_of_data=threshold, concat_answer=True, static_dataset=False, alice=alice)
    all_mlp_vector_with_static, all_attention_vector_with_static, all_mlp_vector_without_static, all_attention_vector_without_static, heads_vectors_with_static, heads_vectors_without_static, all_residual_with_static, all_residual_without_static = MLPCheck.generate_data()



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
    parser.add_argument("--dataset_name", type=str, default="trivia_qa_no_context")

    parser.add_argument("--alice", type=bool, default=False)

    parser.add_argument("--plot_results", type=bool, default=False)
    parser.add_argument("--post_answer", type=bool, default=False)
    parser.add_argument("--pre_answer", type=bool, default=False)
    parser.add_argument("--know_hall_vs_do_not_know_hall_vs_know", type=bool, default=False)
    parser.add_argument("--alice_vs_bad_shot", type=bool, default=False)





    # run dataset creation
    parser.add_argument("--run_dataset_creation", type=bool, default=False,
                        help="run dataset creation - create the hallucination and non-hallucination datasets")
    parser.add_argument("--run_initial_test", type=bool, default=False, help="run initial test on the dataset and create the info for that")

    if parser.parse_args().run_dataset_creation:
        # create dataset
        create_dataset(dataset_name=parser.parse_args().dataset_name,
                       threshold=parser.parse_args().threshold, model_name=parser.parse_args().model_name, alice=parser.parse_args().alice)
    if parser.parse_args().run_initial_test:
        # run initial test
        run_initial_test_on_dataset(threshold=parser.parse_args().threshold, model_name=parser.parse_args().model_name,
                                    dataset_size=parser.parse_args().dataset_size,
                                    dataset_name=parser.parse_args().dataset_name,
                                    concat_answer=parser.parse_args().concat_answer, alice=parser.parse_args().alice)


    if parser.parse_args().plot_results:
        # plot results
        import plot_results
        plot_results.run_plot_results(args_post_answer=parser.parse_args().post_answer,
                                      args_pre_answer=parser.parse_args().pre_answer,
                                      args_type1_vs_type3_know=parser.parse_args().know_hall_vs_do_not_know_hall_vs_know,
                                      args_alice_vs_bad_shot=parser.parse_args().alice_vs_bad_shot,
                                        args_alice=parser.parse_args().alice)
