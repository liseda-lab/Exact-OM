import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from matcha_dl.core.contracts.trainer import EntityMapping, ITrainer

# TODO add validation split


class MLPTrainer(ITrainer):

    def train(
        self,
        epochs: Optional[int] = 50,
        batch_size: Optional[int] = None,
        val_every: Optional[int] = 1,
        save_interval: Optional[int] = 5,
        **kwargs
    ):

        warnings.filterwarnings("ignore", category=UserWarning)

        writer = SummaryWriter(self.logs_dir)

        while self.epoch <= epochs:
            self._model.train()
            _iter = 1

            with tqdm(self._load_data(kind="train", batch_size=batch_size), unit="batch") as tepoch:

                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    self._optimizer.zero_grad()
                    logits = self._model(data)
                    loss = self._loss(logits, target)
                    writer.add_scalar("Loss/train", loss, _iter)

                    loss.backward()
                    self._optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

                    _iter += 1

            if save_interval is not None and save_interval > 0:
                if self.epoch % save_interval == 0:
                    self.save_checkpoint()

            if val_every is not None and val_every > 0 and self.dataset.validation_set is not None:
                if self.epoch % val_every == 0:

                    _ , validation_loss = self.predict(kind='validation')

                    writer.add_scalar("Loss/validation", validation_loss, self.epoch)

                    tepoch.set_postfix(validation_loss=validation_loss)

                    self.log(f"Validation loss at epoch {self.epoch} - {validation_loss}", level="info")

                    if self.stopping is not None:
                        if self.stopping(train_loss=loss.item(), validation_loss=validation_loss):

                            self.log(f"Early stopping at epoch {self.epoch}", level="info")
                            break

            self._epoch += 1

        writer.flush()
        writer.close()

        self.save_checkpoint()

    def predict(self, 
                kind: str = "inference", 
                threshold: Optional[float] = None,
                cardinality: Optional[int] = None,
                **kwargs
    ) -> Tuple[List[EntityMapping], float]:

        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]

        # if supervised use model to calculate scores
        if self.dataset.reference is not None:

            data, target = self._load_data(kind=kind)
            self._model.eval()
            with th.no_grad():
                logits = self._model(data)
                loss = self._loss(logits, target)

            df["Scores"] = logits.cpu()

            return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), loss.item()

        # if unsupervised use max score from matcha
        else:

            df["Scores"] = np.array(df["Features"].values.tolist()).max(axis=1)

            return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), 0.0

    def _load_data(
        self, kind: str = "train", batch_size: Optional[int] = 1
    ) -> DataLoader:

        x = self.dataset.x(kind)
        y = self.dataset.y(kind)

        x = th.tensor(x, dtype=th.float32)
        x = x.to(self.device)

        y = th.tensor(y, dtype=th.float32)
        y = y.unsqueeze(1)
        y = y.to(self.device)

        if kind == "train":
            ds = TensorDataset(x, y)

            return DataLoader(ds, batch_size=batch_size, shuffle=True)

        return x, y
