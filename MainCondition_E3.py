from DiffusionFreeGuidence.TrainCondition_E3 import train, eval
import torch
import os
import torch.multiprocessing as mp

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 4,
        "T": 500,
        "channel": 64,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8,
        "seed": 1234
    }
    if model_config is not None:
        modelConfig = model_config
    
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60001'

    modelConfig["batch_size"] = modelConfig["batch_size"] // n_gpus
    # print('Total batch size:', modelConfig["batch_size"])
    print('Batch size per GPU :', modelConfig["batch_size"])

    mp.spawn(train, nprocs=n_gpus, args=(n_gpus, modelConfig))
    # if modelConfig["state"] == "train":
    #     train(modelConfig)
    # else:
    #     eval(modelConfig)


if __name__ == '__main__':
    main()
