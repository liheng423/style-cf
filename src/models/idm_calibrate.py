import torch
from torch import nn
from torch import Tensor
from torch.utils import data
from tensordict import TensorDict

def evaluate_recursive(model: nn.Module, dataloader: data.DataLoader, criterion: nn.Module, simulator, config: dict):
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
            
            x: TensorDict; y: TensorDict = x.to(device), y.to(device)

            y_self, y_leader = y

            pred_self = simulator.predict(x, y_self, y_leader, pred_func, mask)
            
            loss = criterion(pred_self, y_self)

            running_loss += loss.item()

    return running_loss / num_batches

def _fitness_function(params, idm_model, dataloader, config):
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

def calibrate_idm_genetic(dataloader, idm_model: IDM, calibration_config):
    


    bounds = [(15, 30), (0, 3), (0.5, 3), (0.5, 4), (0.5, 4)]  # 对应 [v0, s0, T, a, b]


    ga = GA(func=lambda params: _fitness_function(params, idm_model, dataloader, calibration_config), 
            n_dim=5,  # 参数维度
            size_pop=10,  # 种群大小
            max_iter=50,  # 最大迭代次数
            prob_mut=0.2,  # 变异概率
            lb=[bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0], bounds[4][0]],  # 参数下界
            ub=[bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1], bounds[4][1]],  # 参数上界
            precision=1e-2)  # 精度
    
    ga.to(calibration_config["device"])


    best_params, best_loss = ga.run()

    
    return best_params, best_loss