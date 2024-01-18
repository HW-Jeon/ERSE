import os

import torch

import config_TransEE as cfgs
import utilities as util
from openke.config import Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model.TransEE import TransEE


def setThreshold(strDataset, mode, ent_type):
    if strDataset == "FB15K":
        cfgs.entropy_normal_min = 0.625

    elif strDataset == "FB15K237":
        if mode == "MINMAX":
            if "n_shannon" in type_entropy:
                cfgs.entropy_normal_min = 2.8125
            elif "c_shannon" in type_entropy:
                cfgs.entropy_normal_min = 3.125
            elif "n_renyi" in ent_type:
                cfgs.entropy_normal_min = 2.4219
            else:
                cfgs.entropy_normal_min = 2.8125

        else:
            if "n_shannon" in type_entropy:
                cfgs.entropy_normal_min = 3.125
            elif "c_shannon" in type_entropy:
                cfgs.entropy_normal_min = 3.125
            elif "n_renyi" in ent_type:
                cfgs.entropy_normal_min = 2.1875
            else:
                cfgs.entropy_normal_min = 2.1875


if __name__ == "__main__":
    # print(util.get_csv_path_short())
    # input()

    # cfgs.default_entropy_dir_path = "./csv/FB15K237/GOOD_PERFOMANCE/PDF_Categorical_Mixed/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_05_075/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_025_05/entropy_k_"
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_ks_eh_Mixed_0.75_0.5/entropy_k_"

    # cfgs.setDataset("WN18RR")
    # cfgs.default_entropy_dir_path = "./csv/FB15K237/PDF_Categorical_TRUE_Mixed_0.5_0.75/entropy_k_"

    strDataset = "FB15K"
    # strDataset = "FB15K237"
    cfgs.default_entropy_dir_path = (
        # "./csv/FB15K237/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        f"./csv/{strDataset}/FINAL_N_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/PER_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/TOT_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_NP_PDF_Trained_0.5_0.75/entropy_k_"
        # "./csv/FB15K237/FINAL_PDF_PAIRED_Trained_0.5_0.75/entropy_k_"
    )

    # strDataset = "WN18RR"
    # cfgs.default_entropy_dir_path = (
    #     # f"./csv/{strDataset}/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    #     f"./csv/{strDataset}/DROP_RES_PDF_Categorical_Trained_0.5_0.75/entropy_k_"
    # )

    cfgs.setDataset(strDataset)

    if "WN" in strDataset:
        cfgs.hit_k_limits = [3]

    # /home/kist/workspace/OpenKE/csv/FB15K/DEFAULT_RES_PDF_Categorical_Trained_0.5_0.75
    # # dataloader for training
    # train_dataloader = TrainDataLoader(
    #     in_path="./benchmarks/FB15K237/",
    #     nbatches=100,
    #     threads=8,
    #     sampling_mode="normal",
    #     bern_flag=1,
    #     filter_flag=1,
    #     neg_ent=25,
    #     neg_rel=0,
    # )

    # # dataloader for test
    # test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    testpaths = ["_11", "_1n", "_n1", "_nn"]
    # testpaths = ["_n1"]

    # testpaths = ["_n1", "_nn"]
    # testpaths = ["_11", "_1n"]
    basicPath = f"./benchmarks/{strDataset}"

    # os.renames(f"{basicPath}/Test.txt", f"{basicPath}/Test_testing.txt")

    os.renames(f"{basicPath}/test2id.txt", f"{basicPath}/test2id_testing.txt")

    results = {}

    for objset in testpaths:
        result = {}

        os.renames(f"{basicPath}/test2id{objset}.txt", f"{basicPath}/test2id.txt")

        train_dataloader = util.dataLoader(strDataset)

        test_dataloader = TestDataLoader(f"./benchmarks/{strDataset}/", "link")

        # print(test_dataloader.testTotal)
        # input()

        # define the model
        transee = TransEE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
        )

        # for path_id in util.get_csv_path():
        #     cfgs.entropy_path_id = path_id

        # cfgs.WRITE_EVAL_RESULT = True

        # for eval_norm_mode in ["LOGIT"]:

        # cfgs.num_count_threshold = -1

        for eval_norm_mode in ["MINMAX", "LOGIT"]:
            cfgs.MODE_EVAL_NORM = eval_norm_mode

            for ths in cfgs.num_count_thresholds:
                cfgs.num_count_threshold = ths

                for path_id in util.get_csv_path_short():
                    cfgs.entropy_path_id_short = path_id

                    for type_entropy in cfgs.column_List_entropy_MINIMUM:
                        # for type_entropy in ["num_testing"]:

                        cfgs.reverse_flag = False
                        # cfgs.reverse_flag = True

                        setThreshold(strDataset, eval_norm_mode, type_entropy)

                        cfgs.types_of_entropy = type_entropy

                        util.print_eval_header()

                        # if "diff" in type_entropy:
                        #     cfgs.reverse_flag = True

                        # else:
                        #     cfgs.reverse_flag = False

                        if cfgs.WRITE_EVAL_RESULT:
                            util.endl(
                                f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}] - EVAL_RESULT Writing"
                            )

                        else:
                            util.endl(
                                f"[{ths} - {path_id} {type_entropy} {cfgs.reverse_flag}]"
                            )

                        tester = Tester(
                            model=transee, data_loader=test_dataloader, use_gpu=True
                        )
                        (mrr, mr, hit10, hit3, hit1) = tester.run_link_prediction(
                            type_constrain=False
                        )

                        # result[f"{eval_norm_mode}-{type_entropy}"] = (
                        #     mrr,
                        #     mr,
                        #     hit10,
                        #     hit3,
                        #     hit1,
                        # )

                        result[f"{eval_norm_mode}-{type_entropy}"] = hit10

                        print(result[f"{eval_norm_mode}-{type_entropy}"])
                        # input()

                        del tester
                        torch.cuda.empty_cache()

                        if cfgs.num_count_threshold < 0:
                            break
                    if cfgs.num_count_threshold < 0:
                        break

                    cfgs.WRITE_EVAL_RESULT = False
        results[objset] = result

        os.renames(f"{basicPath}/test2id.txt", f"{basicPath}/test2id{objset}.txt")

    os.renames(f"{basicPath}/test2id_testing.txt", f"{basicPath}/test2id.txt")

    print("Final results:")
    print(results)
