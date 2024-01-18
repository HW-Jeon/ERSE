import torch

import openke
from openke.config import Tester, Trainer
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransD, TransE, TransH, TransR
from openke.module.strategy import NegativeSampling

strDataset = "FB15K237"

def TransEs():
    
    global strDataset
    
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=1.0,
        use_gpu=True,
    )
    trainer.run()
    transe.save_checkpoint(f"./basemodel/{strDataset}/transe.ckpt")

    # test the model
    transe.load_checkpoint(f"./basemodel/{strDataset}/transe.ckpt")
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transe.save_parameters(f"./basemodel/{strDataset}/transe.json")


def TransHs():
    global strDataset
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transh,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=0.5,
        use_gpu=True,
    )
    trainer.run()
    transh.save_checkpoint(f"./basemodel/{strDataset}/transh.ckpt")

    # test the model
    transh.load_checkpoint(f"./basemodel/{strDataset}/transh.ckpt")
    tester = Tester(model=transh, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transh.save_parameters(f"./basemodel/{strDataset}/transh.json")


def TransRs():
    global strDataset
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path="./benchmarks/FB15K237/", sampling_mode="link"
    )

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True,
    )

    model_e = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
    )

    model_r = NegativeSampling(
        model=transr,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    # pretrain transe
    trainer = Trainer(
        model=model_e,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=0.5,
        use_gpu=True,
    )
    trainer.run()
    parameters = transe.get_parameters()
    transe.save_parameters(f"./basemodel/{strDataset}/transr_transe.json")

    # train transr
    transr.set_parameters(parameters)
    trainer = Trainer(
        model=model_r,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=1.0,
        use_gpu=True,
    )
    trainer.run()
    transr.save_checkpoint(f"./basemodel/{strDataset}/transr.ckpt")

    # test the model
    transr.load_checkpoint(f"./basemodel/{strDataset}/transr.ckpt")
    tester = Tester(model=transr, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    parameters = transr.get_parameters()
    transr.save_parameters(f"./basemodel/{strDataset}/transr.json")


def TransDs():
    global strDataset
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=256,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transd,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=1.0,
        use_gpu=True,
    )
    trainer.run()
    transd.save_checkpoint(f"./basemodel/{strDataset}/transd.ckpt")

    # test the model
    transd.load_checkpoint(f"./basemodel/{strDataset}/transd.ckpt")
    tester = Tester(model=transd, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transd.save_parameters(f"./basemodel/{strDataset}/transd.json")


if __name__ == "__main__":
    TransEs()
    torch.cuda.empty_cache()
    TransHs()
    torch.cuda.empty_cache()
    TransDs()
    torch.cuda.empty_cache()
    TransRs()
    torch.cuda.empty_cache()
