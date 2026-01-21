
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensordict import TensorDict
from src.models import utils

from src.models.filters import CFFilter
from src.models.style_cf import StyleTransformer, batch_apply, reaction_time, time_headway, transformer_mask
from src.models.utils import SampleDataPack
from src.models import dataset
from src.schema import CFNAMES as CF
from src.models.configs import style_train_config
from src.utils.logger import logger


### Machine Learning Model ###

def build_dataset(d: SampleDataPack, d_filters: List, d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for car-following model training.
    """
    d = d.normalize_kilopost()  # normalize KILO to rising
    d.append_col(d[:, :, CF.LEAD_V] - d[:, :, CF.SELF_V], CF.DELTA_V)
    d.append_col(d[:, :, CF.LEAD_X] - d[:, :, CF.SELF_X], CF.DELTA_X)
    d_filter = CFFilter(d, d_filter_config)
    if d_filters and isinstance(d_filters[0], str):
        d_filters = [getattr(d_filter, name) for name in d_filters]
    d = d_filter.filter(d_filters)

    d.force_consistent()

    

    return d

def build_style_dataset(
    d: SampleDataPack,
    d_filters: List[CFFilter],
    d_filter_config: dict,
    data_config: dict | None = None,
    seed: int = 42,
) -> tuple[SampleDataPack, DataLoader, DataLoader, list]:
    """
    Build the dataset for style-based car-following model training.
    """
    d = build_dataset(d, d_filters, d_filter_config)
    
    # construct new features
    d.append_col(
        np.expand_dims(
            np.tile(np.arange(0, 30, 0.1), (len(d[:, 0, CF.LEAD_V]), 1)),
            2,
        ),
        CF.TIME,
    )
    d.append_col(
        batch_apply(
            reaction_time,
            [d[:, :, CF.LEAD_V], d[:, :, CF.SELF_V], d[:, :, CF.TIME]],
        )[:, :, np.newaxis],
        CF.REACT,
    )
    d.append_col(
        batch_apply(
            time_headway,
            [d[:, :, CF.DELTA_X] - d[:, :, CF.LEAD_L], d[:, :, CF.SELF_V]], # use net distance
        ),
        CF.THW,
    )

    if data_config is None:
        return d
    train_loader, test_loader, scalers = pipeline(d, data_config, seed)
    return d, train_loader, test_loader, scalers


def pipeline(d: SampleDataPack, data_config: dict, seed: int) -> tuple[DataLoader, DataLoader, list]:
    x_groups = data_config["x_groups"]
    y_groups = data_config["y_groups"]

    x_data = [d[:, :, group["features"]] for group in x_groups.values()]
    y_data = [d[:, :, group["features"]] for group in y_groups.values()]

    num_samples = y_data[0].shape[0]
    indices = np.arange(num_samples)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - data_config["train_data_ratio"],
        random_state=seed,
    )

    x_train = [xi[train_idx] for xi in x_data]
    x_test = [xi[test_idx] for xi in x_data]
    y_train = [yi[train_idx] for yi in y_data]
    y_test = [yi[test_idx] for yi in y_data]

    scalers = {}
    for key, group in x_groups.items():
        if not group.get("transform", True):
            continue
        scalers[key] = data_config["scaler"]()

    for data_idx, key in enumerate(x_groups.keys()):
        if key not in scalers:
            continue
        scalers[key] = dataset._fit_scaler(scalers[key], x_train[data_idx])
    
    transform = dataset.make_transform(scalers, x_groups)

    dataset_cls = data_config["dataset"]
    train_dataset = dataset_cls(*x_train, *y_train, data_config=data_config, transform=transform)
    test_dataset = dataset_cls(*x_test, *y_test, data_config=data_config, transform=transform)
    batch_size = data_config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=utils._collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=utils._collate,
    )
    return train_loader, test_loader, scalers

@logger.decorator("train_stylecf")
def train_stylecf(model_config, train_config, train_loader: DataLoader, test_loader: DataLoader):
    
    
    model = model_config["model_name"](model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)
    
    
    optim_func = train_config["optim"]
    criterion = train_config["loss_func"]
    
    train_losses = []
    val_losses = []

    num_epoch = train_config["num_epoch"]

    model.use_dummy_style = False

    logger.info(f"Starting Style-cf Training for {num_epoch} epochs on {device}")
    logger.info(f"Model: {model}")
    logger.info(f"Optimizer: {optim_func}")
    logger.info(f"Loss Function: {criterion}")
    logger.info(f"Using Style: {not model.use_dummy_style}")



    # ======== START OF TRAINING ======== #
    # STAGE 1
    assert isinstance(model, StyleTransformer)

    # for param in model.embedder.parameters():
    #     param.requires_grad = False

    optimizer = optim_func(model.parameters(), lr=1e-4)

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',       
        factor=1e-1,      
        patience=5,       
        verbose=True     
    )


        
    def _train_loop(model, train_loader, test_loader, optimizer, criterion, num_epochs):

        best_val_loss = float('inf')  
            
        for epoch in range(num_epochs):

            # train_sampler.set_epoch(epoch)

            # train_loss = train_recursive(model, train_loader, criterion, optimizer, multi_agents, train_config)
            # val_loss = evaluate_recursive(model, test_loader, criterion, multi_agents, train_config)
            train_loss = train(model, train_loader, criterion, optimizer, train_config)
            val_loss = evaluate(model, test_loader, criterion, train_config)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            ## Learning rate ##

            scheduler.step(val_loss) 
            current_lr = optimizer.param_groups[0]['lr']


            ### Model Saving ###

            # Save model if the validation loss is the best so far
            if epoch >= 5 and val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model with the best validation loss
                utils.model_save(model, train_config["best_model_path"])
                print(f"Saved Best Model at Epoch {epoch+1} with Val Loss: {val_loss:.4f}")

    return _train_loop(model, train_loader, test_loader, 
    optimizer, criterion, num_epoch)    


def train(model, dataloader, criterion, optimizer, train_config):

    device = train_config["device"]
    max_norm = train_config["max_norm"]
    dt = train_config["dt"]


    model.train()
    epoch_loss = 0.0
    num_batches = len(dataloader)

    for x, y in tqdm(dataloader):
        # move to device

        x, y = x.to(device), y.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y, dt)  # dt = 0.1
        
        # 反向传播和优化
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / num_batches

def evaluate(model, dataloader, criterion, train_config):

    device = train_config["device"]
    dt = train_config["dt"]
    

    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for x, y in dataloader:
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)

            # 前向传播
            outputs = model(x)
            loss = criterion(outputs, y, dt)

            running_loss += loss.item()

    return running_loss / num_batches
