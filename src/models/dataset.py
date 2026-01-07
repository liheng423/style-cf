import torch
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict

class TransformerDataset(Dataset):
    def __init__(self, x_seq_enc=None, x_seq_dec=None, x_static=None, y_seq=None, y_static=None, data_config=None):
        self.x_seq_enc = torch.tensor(x_seq_enc).float()
        self.x_seq_dec = torch.tensor(x_seq_dec).float()
        self.x_static = torch.tensor(x_static).float() if x_static is not None else None
        self.y_seq = torch.tensor(y_seq).float()
        self.y_static = torch.tensor(y_static).float() if y_static is not None else None

        self.num_samples = self.x_seq_enc.shape[0]
        self.total_len = self.x_seq_enc.shape[1]

        self.seq_len = data_config["seq_len"]
        self.label_len = data_config["label_len"]
        self.pred_len = data_config["pred_len"]
        self.stride = data_config.get("stride", 1) if data_config else 1

        self.indices = [
            (i, t)
            for i in range(self.num_samples)
            for t in range(0, self.total_len - (self.seq_len + self.pred_len) + 1, self.stride)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - tuple: (encoder_input, decoder_input, static_features_x)
                - tuple: (sequence_output, static_features_y)
        """
        i, t = self.indices[idx]

        # Encoder input: sequence from time t to t + seq_len
        x_enc = self.x_seq_enc[i, t : t + self.seq_len, :]

        # Decoder input construction
        # The window covers the label_len part (known history) and the pred_len part (to be predicted)
        window = self.x_seq_dec[i, t + self.seq_len - self.label_len : t + self.seq_len + self.pred_len, :]
        label_part = window[:self.label_len]
        pred_part = window[self.label_len:].clone()
        pred_part[:, 0] = label_part[:, 0].mean() # For the first feature in the prediction part, use the mean of the label part
        x_dec = torch.cat([label_part, pred_part], dim=0)

        # Static features and target sequences
        x_static = self.x_static[i] if self.x_static is not None else None
        y_seq = self.y_seq[i, t + self.seq_len - 1 : t + self.seq_len + self.pred_len, :]
        y_static = self.y_static[i] if self.y_static is not None else None

        return (x_enc, x_dec, x_static), (y_seq, y_static)

class StyledTransfollowerDataset(TransformerDataset):
    def __init__(self, x_seq_enc, x_seq_dec, x_style, y_seq, data_config=None):
        super().__init__(x_seq_enc, x_seq_dec, None, y_seq, None, data_config)
        self.x_style = x_style

    def __getitem__(self, idx):
        (x_enc, x_dec, x_static), (y_seq, y_static) = super().__getitem__(idx)
        i, t = self.indices[idx]

        # 样本 style: 从开头到 t + seq_len 的 window
        x_style = self.x_style[i, t: t + self.seq_len, :]

        x = x_enc, x_dec, x_style
        y = y_seq
        return x, y