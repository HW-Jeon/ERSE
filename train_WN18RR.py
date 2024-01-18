import torch

import openke
from openke.config import Tester, Trainer
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransD, TransE, TransH, TransR
from openke.module.strategy import NegativeSampling


def TransEs(dim_size):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18RR/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim_size,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=6.0),
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
    transe.save_checkpoint("./checkpoint/WN18RR/transe.ckpt")

    # test the model
    transe.load_checkpoint("./checkpoint/WN18RR/transe.ckpt")
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transe.save_parameters("./result/WN18RR/transe.json")


def TransHs(dim_size):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18RR/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim_size,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transh,
        loss=MarginLoss(margin=6.0),
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
    transh.save_checkpoint("./checkpoint/WN18RR/transh.ckpt")

    # test the model
    transh.load_checkpoint("./checkpoint/WN18RR/transh.ckpt")
    tester = Tester(model=transh, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transh.save_parameters("./result/WN18RR/transh.json")


def TransRs(dim_size):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18RR/",
        nbatches=512,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # dataloader for test
    test_dataloader = TestDataLoader(in_path="./benchmarks/WN18RR/", sampling_mode="link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim_size,
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
        dim_e=dim_size,
        dim_r=dim_size,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
    )

    model_r = NegativeSampling(
        model=transr,
        loss=MarginLoss(margin=6.0),
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
    transe.save_parameters("./result/WN18RR/transr_transe.json")

    # train transr
    transr.set_parameters(parameters)
    trainer = Trainer(
        model=model_r,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=0.1,
        use_gpu=True,
    )
    trainer.run()
    transr.save_checkpoint("./checkpoint/WN18RR/transr.ckpt")

    # test the model
    transr.load_checkpoint("./checkpoint/WN18RR/transr.ckpt")
    tester = Tester(model=transr, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    parameters = transr.get_parameters()
    transr.save_parameters("./result/WN18RR/transr.json")


def TransDs(dim_size):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18RR/",
        nbatches=256,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
    )

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=dim_size,
        dim_r=dim_size,
        p_norm=1,
        norm_flag=True,
    )

    # define the loss function
    model = NegativeSampling(
        model=transd,
        loss=MarginLoss(margin=8.0),
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
    transd.save_checkpoint("./checkpoint/WN18RR/transd.ckpt")

    # test the model
    transd.load_checkpoint("./checkpoint/WN18RR/transd.ckpt")
    tester = Tester(model=transd, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

    transd.save_parameters("./result/WN18RR/transd.json")


if __name__ == "__main__":
    dim = 100

    TransEs(dim)
    torch.cuda.empty_cache()
    TransHs(dim)
    torch.cuda.empty_cache()
    TransRs(dim)
    torch.cuda.empty_cache()
    TransDs(dim)
    torch.cuda.empty_cache()
