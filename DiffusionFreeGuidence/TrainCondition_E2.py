

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition_E2 import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition_E2 import UNet
from Scheduler import GradualWarmupScheduler

from torch.utils.data import DataLoader
from data_utils import DistributedBucketSampler
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import customAudioDataset as data
# from DiffusionFreeGuidence.TrainCondition_E1 import train, eval
# def main(params):
#     """Assume Single Node Multi GPUs Training Only"""
#     assert torch.cuda.is_available(), "CPU training is not allowed."

#     n_gpus = torch.cuda.device_count()
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '60001'

#     batch_size = params.batch_size // n_gpus
#     print('Total batch size:', params.batch_size)
#     print('Batch size per GPU :', batch_size)

#     mp.spawn(run, nprocs=n_gpus, args=(n_gpus,))
def fix_len_compatibility(length, num_downsamplings_in_unet=4):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
def collate_fn(batch):
    B = len(batch)
    
    # rpsody: [1, 329, 1024]
    # timbre: [1, 512]
    # target: [1, 329, 128]

    max_length = max([item[1].shape[-2] for item in batch])
    max_length = fix_len_compatibility(max_length)
    # content_max_length = max([item[3].shape[-2] for item in batch])

    pro_nfeats = batch[0][1].shape[-1]
    tim_nfeats = 512 # batch[0][2].shape[-1]
    con_nfeats = 1024
    tar_nfeats = batch[0][4].shape[-1]

    # print(type(B), type(pro_max_length), type(pro_nfeats))
    pro = torch.zeros((B, max_length, pro_nfeats), dtype=torch.float32)
    tar = torch.zeros((B, max_length, tar_nfeats), dtype=torch.float32)
    tim = torch.zeros((B, max_length, tim_nfeats), dtype=torch.float32)
    con = torch.zeros((B, max_length, con_nfeats), dtype=torch.float32)
    lengths = []

    for i, item in enumerate(batch):
        pro_, tim_, con_, tar_ = item[1], item[2], item[3], item[4]
        # if target.size()[-2] > 1000:
        #     # print('cut', target.size())
        #     start = random.randint(0, target.size()[1]-self.tensor_cut-1)
        #     target = target[:, start:start+self.tensor_cut,:]
        #     # print(target.size())
        #     self.lengths[idx] = self.tensor_cut

        lengths.append(pro_.shape[-2])
        # tar_lengths.append(tar_.shape[-2])

        pro[i,:pro_.shape[-2],:] = pro_
        tar[i,:tar_.shape[-2],:] = tar_
        con[i,:con_.shape[-2],:] = con_
        tim[i,:tar_.shape[-2],:] = tim_

    # tim = tim.expand(-1, )
    lengths = torch.LongTensor(lengths)
    # tar_lengths = torch.LongTensor(tar_lengths)

    # print(pro.shape, tim.shape, tar.shape, lengths)
    # print(tim)
    # labels = {
    #     'pro': pro,
    #     'tim': tim,
    #     'con': con
    # }
    # data = {
    #     'tar': tar, 
    #     'lengths': lengths
    # }
    # pro, tim, con, tar, lengths
    return pro, tim, con, tar, lengths

# def run(rank, n_gpus):
    

def train(rank, n_gpus, modelConfig: Dict):
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(modelConfig["seed"])
    np.random.seed(modelConfig["seed"])
    
    if rank == 0:
        print('Set devices ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if n_gpus > 1:
        device = torch.device("cuda:{:d}".format(rank))

    if rank == 0:
        print('Initializing logger...')
    
    # logger = SummaryWriter(log_dir=log_dir)
    
    if rank == 0:
        print('Initializing data loaders...')
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = data.CustomAudioDataset('train_E3.txt', tensor_cut=200)
    train_sampler = DistributedBucketSampler(
        # logger,
        dataset,
        modelConfig["batch_size"],
        # [0, 100, 200, 300, 400, 500, 600,700,800, 900, 1000,2500],
        [0, 50, 100, 150, 200, 300, 400,500,600, 700, 800,2500],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn,)
    dataloader = DataLoader(dataset=dataset,
                        collate_fn=collate_fn,
                        num_workers=8, shuffle=False, batch_sampler=train_sampler)
    
    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    net_model.cuda(rank)
    # print(net_model)
    net_model = DDP(net_model, device_ids=[rank],find_unused_parameters=True)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).cuda(rank)

    # start training
    for e in range(modelConfig["epoch"]):
        dataloader.batch_sampler.set_epoch(e)
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for pro, tim, con, tar, lengths in tqdmDataLoader:
                # train
                images = tar
                labels = {
                    'pro': pro.cuda(rank),
                    'tim': tim.cuda(rank),
                    'con': con.cuda(rank)
                }
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.cuda(rank)
                # labels = labels.to(device) + 1
                # if np.random.rand() < 0.1:
                #     labels = torch.zeros_like(labels).to(device)
                # loss = trainer(x_0, labels).sum() / b ** 2. # labels - emb
                loss = trainer(x_0, labels).sum() / images.shape[1]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        # print("labels: ", labels)
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

# if __name__ == "__main__":
#     main(params)