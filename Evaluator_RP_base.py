import os

import torch

import config_TransEE as cfgs
import utilities as util
from openke.config import Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model.TransEE import TransEE

if __name__ == "__main__":
    strDataset = "FB15K"
    cfgs.default_entropy_dir_path = (
        f"./csv/{strDataset}/FINAL_N_PDF_Trained_0.5_0.75/entropy_k_"
    )

    cfgs.setDataset(strDataset)

    models = util.load_models_list(tags=f"/Pre_{strDataset}", devices=cfgs.devices)

    testpaths = ["_11", "_1n", "_n1", "_nn"]
    # testpaths = ["_n1"]

    # testpaths = ["_n1", "_nn"]
    # testpaths = ["_11", "_1n"]
    basicPath = f"./benchmarks/{strDataset}"

    os.renames(f"{basicPath}/test2id.txt", f"{basicPath}/test2id_testing.txt")

    results = {}

    for objset in testpaths:
        os.renames(f"{basicPath}/test2id{objset}.txt", f"{basicPath}/test2id.txt")

        test_dataloader = TestDataLoader(f"./benchmarks/{strDataset}/", "link")

        for path_id in util.get_csv_path_short():
            cfgs.entropy_path_id_short = path_id
            result = {}
            for strModel in cfgs.strModels:
                tester = Tester(
                    model=models[strModel], data_loader=test_dataloader, use_gpu=True
                )
                tester.run_link_prediction(type_constrain=False)
                (mrr, mr, hit10, hit3, hit1) = tester.run_link_prediction(
                    type_constrain=False
                )

                result[f"{strModel}"] = hit10

        results[objset] = result

        os.renames(f"{basicPath}/test2id.txt", f"{basicPath}/test2id{objset}.txt")

    os.renames(f"{basicPath}/test2id_testing.txt", f"{basicPath}/test2id.txt")

    print("Final results:")
    print(results)
