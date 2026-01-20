import random
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data
from typing import Dict
import torch
from torch import nn
from torch import Tensor
from torch.utils import data
from tensordict import TensorDict
from tqdm import tqdm
from src.models.benchmarks import IDM
from src.models.agent import Agent
from sko.GA import GA
from src.models.dataset import IDMDataset
from src.models.utils import SampleDataPack, ensure_dir
from src.models.model_trainer import build_dataset
from src.schema import CFNAMES as CF

def evaluate_recursive(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, simulator: Agent, config: dict):
    """
    The difference is that this function takes a time series instead of a set of time instances,
    the function can evaluate the fitness of the whole trajectory using recursive evaluation step by step.
    """

    device = config["device"]
    pred_func = config["pred_func"]
    mask = config["mask"]

    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():

        for x, y in dataloader:
            
            x= x.to(device)

            y_self, y_leader = y
            y_self = y_self.to(device)
            y_leader = y_leader.to(device)


            pred_self = simulator.predict(x, y_self, y_leader, pred_func, mask)
            
            loss = criterion(pred_self, y_self)

            running_loss += loss.item()

    return running_loss / num_batches

def _fitness_function(params, idm_model: IDM, dataloader: data.DataLoader, config):
    """
    Calculate the fitness of a set of parameters for the IDM model.
    
    Parameters:
    - params: list of parameters for the IDM model (v0, s0, T, a, b).
    - idm_model: the IDM model class.
    - idm_data: the dataset for training.
    - loss_func: the loss function used to calculate the error.
    - device: device to run the model (CPU or GPU).
    - time_interval: time interval for the model prediction.
    
    Returns:
    - float: the loss value, which will be minimized by GA.
    """
    device = config["device"]
    loss_func = config["loss"]
    dt = config["downsample"]
    scaler = config["scaler"]
    start_step = int(config["start_step"] / dt)
    update_func = config["update_func"]
    pred_horizon = int(config["pred_horizon"] / dt)
    historic_step = int(config["historic_step"] / dt)

    model = idm_model(params, use_torch=True).to(device)
    simulator = Agent(model, dt, pred_horizon, historic_step, scaler, start_timestep=start_step)
    simulator._update_train_series = update_func(simulator)

    fitness = evaluate_recursive(model, dataloader, loss_func, simulator, config)

    return fitness

def calibrate_idm_genetic(dataloader, idm_model: IDM, config: dict):

    bounds = [(15, 30), (0, 3), (0.5, 3), (0.5, 4), (0.5, 4)]  # 对应 [v0, s0, T, a, b]
    ga = GA(func=lambda params: _fitness_function(params, idm_model, dataloader, config), 
            n_dim=5,  # 参数维度
            size_pop=10,  # 种群大小
            max_iter=50,  # 最大迭代次数
            prob_mut=0.2,  # 变异概率
            lb=[bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0], bounds[4][0]],  # 参数下界
            ub=[bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1], bounds[4][1]],  # 参数上界
            precision=1e-2)  # 精度
    
    ga.to(config["device"])


    best_params, best_loss = ga.run()

    
    return best_params, best_loss


def calibrate_idm(idm: nn.Module, id_datapack: Dict[int, SampleDataPack], config):
    """
        Calibrate the IDM model using genetic algorithm. Note that each ID is seperately calibrated, and the final results take the average of the parameters. 
    
    """


    sample_size = 1000  
    random.seed(42)  
    sample_indices = random.sample(range(len(id_datapack)), sample_size)
    results = []

    for idx in tqdm(sample_indices):
        idm_dataset = IDMDataset(id_datapack[idx][:, :, config["features"]], 
                                    id_datapack[idx][:, :, [CF.SELF_X, CF.SELF_V, CF.SELF_A]], 
                                    id_datapack[idx][:, :, [CF.LEAD_X, CF.LEAD_V, CF.LEAD_A]], 
                                    int(config["downsample"] / config["resolution"]))
        idm_dataloader = DataLoader(idm_dataset, 1, collate_fn=lambda x: x[0])

        best_params, best_loss = calibrate_idm_genetic(idm_dataloader, idm, config)

        results.append({
            'ID': id_datapack[idx][0, 0, CF.SELF_ID],
            'v0': best_params[0],
            's0': best_params[1],
            'T': best_params[2],
            'a': best_params[3],
            'b': best_params[4],
            'best_loss': best_loss[0]
        })

    df = pd.DataFrame(results)
    ensure_dir(idm_config["save_path"])
    df.to_csv(idm_config["save_path"], index=False)

    print(f"Results saved to {idm_config['save_path']}")