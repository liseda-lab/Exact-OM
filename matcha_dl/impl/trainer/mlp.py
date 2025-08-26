import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from matcha_dl.core.contracts.trainer import EntityMapping, ITrainer
from matcha_dl.core.entities.dataset import DatasetMask

# TODO add validation split


class MLPTrainer(ITrainer):

    def train(
        self,
        epochs: int = 50,
        batch_size: Optional[int] = None,
        val_every: Optional[int] = 1,
        save_interval: Optional[int] = 5,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        
        if self.skip_training:
            self.log("Skipping training, because checkpoint already exists", level="info")
            return

        warnings.filterwarnings("ignore", category=UserWarning)

        writer = SummaryWriter(self.logs_dir)
        early_stopping = False

        self.dataset.default_kind = DatasetMask.train

        while self.epoch <= epochs and not early_stopping:
            self.model.train()
            _iter = 1

            with tqdm(DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), unit="batch") as tepoch:

                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {self.epoch}")

                    data, target = data.to(self.device), target.unsqueeze(1).to(self.device)

                    self._optimizer.zero_grad()
                    logits = self.model(data)
                    loss = self._loss(logits, target)
                    writer.add_scalar("Loss/train", loss, _iter)

                    loss.backward()
                    self._optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

                    _iter += 1

                if val_every is not None and val_every > 0 and self.dataset.validation_set is not None:
                    if self.epoch % val_every == 0:

                        _ , validation_loss = self.predict(kind=DatasetMask.validation)

                        writer.add_scalar("Loss/validation", validation_loss, self.epoch)

                        tepoch.set_postfix(validation_loss=validation_loss)

                        self.log(f"Validation loss at epoch {self.epoch} - {validation_loss}", level="info")

                        if self.stopping is not None:
                            if self.stopping(validation_loss=validation_loss):
                                self.log(f"Early stopping at epoch {self.epoch}", level="info")
                                early_stopping = True
                                break

            if save_interval is not None and save_interval > 0:
                if self.epoch % save_interval == 0:
                    self.save_checkpoint()

            self._epoch += 1

        writer.flush()
        writer.close()

        self.save_checkpoint()

    def predict(self, 
                kind: DatasetMask = DatasetMask.inference,
                batch_size: int = 32,
                threshold: Optional[float] = None,
                cardinality: Optional[int] = None,
                **kwargs
    ) -> Tuple[List[EntityMapping], float]:
        
        self.dataset.default_kind = kind

        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]

        # if supervised use model to calculate scores
        if self.dataset.reference is not None:

            dataloader = DataLoader(self.dataset, batch_size=1024, shuffle=False)

            self.model.eval()
            all_logits = []
            total_loss = 0.0

            with th.no_grad():

                for data, target in dataloader:
                    data, target = data.to(self.device), target.unsqueeze(1).to(self.device)

                logits = self.model(data)
                loss = self._loss(logits, target)
                total_loss += loss.item() * data.size(0)
                all_logits.append(logits.cpu())

            df["Scores"] = th.cat(all_logits).numpy()
            avg_loss = total_loss / len(self.dataset)

            return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), avg_loss

        # if unsupervised use max score from matcha
        else:

            df["Scores"] = np.array(df["Features"].values.tolist()).max(axis=1)

            return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), 0.0
        
    def apply_prefilter(self,
                        threshold: Optional[float] = None,
                        cardinality: Optional[int] = None,
                        **kwargs
    ) -> List[EntityMapping]:
        """
        Apply prefiltering to the dataset based on the features.
        """
        
        df = self.dataset.dataframe.copy()
        df = df[df[DatasetMask.prefiltered] == True]

        if df.empty:
            self.log("No data to prefilter", level="warning")
            return []
        
        df["Scores"] = np.array(df["Features"].values.tolist()).max(axis=1)
        return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality)




