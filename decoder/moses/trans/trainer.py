import torch
import torch.optim as optim
from tqdm.auto import tqdm

from torch.nn.utils import clip_grad_norm_

from moses.interfaces import MosesTrainer
from moses.utils import (
    SmilesOneHotVocab,
    Logger,
    smiles_tokenize,
    set_torch_seed_to_all_gens,
    compute_reconstruction_trans
)
from moses.trans.misc import CosineAnnealingLRWithRestart, KLAnnealer
#from tensorboardX import SummaryWriter
import torch.nn.functional as F


class TranslationTrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config
        self.tb_writer = None
        self.collate_fn = None

    def get_vocabulary(self, data):
        return SmilesOneHotVocab.from_data(data, tokenizer=smiles_tokenize)

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def create_tensors(string, device):
            regression_values = torch.tensor(string[2:])
            return [
                model.string2tensor(string[0], device=device),
                model.string2tensor(string[1], device=device),
                regression_values,
            ]

        def collate(data):
            sort_fn = lambda x: len(x[0])
            # data.sort(key=sort_fn, reverse=True)
            tensors = [create_tensors(string, device) for string in data]
            tensors.sort(key=sort_fn, reverse=True)
            return tensors

        if self.collate_fn:
            return self.collate_fn
        else:
            return collate

    def _train_epoch(
        self,
        model,
        regression_model,
        epoch,
        tqdm_data,
        val_loader,
        kl_weight=0,
        optimizer=None,
    ):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        regression_loss_weight = self.config.regression_loss_weight

        num_batches = len(tqdm_data)
        for i, input_batch in enumerate(tqdm_data):
            # input_batch = tuple(data.to(model.device) for data in input_batch)
            # print(i)
            randomized_smiles, canonical_smiles, regression_targets = zip(
                *input_batch
            )
            regression_targets = torch.stack(regression_targets, dim=0).to(
                self.config.device
            )
            # Forward
            mu, logvar, z, kl_loss, recon_loss = model(
                randomized_smiles, canonical_smiles
            )

            predicted = regression_model(z)
            regression_loss = F.mse_loss(predicted, regression_targets)

            loss = (
                kl_weight * kl_loss
                + recon_loss
                + regression_loss_weight * regression_loss
            )

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    self.get_optim_params(model, regression_model),
                    self.config.clip_grad,
                )
                optimizer.step()

            lr = (
                optimizer.param_groups[0]["lr"]
                if optimizer is not None
                else 0.0
            )

            postfix = [
                f"step={i}",
                f"loss={loss:.3f}",
                f"(kl={kl_loss:.5f}",
                f"recon={recon_loss:.5f})",
                f"regr={regression_loss:.5f})",
                f"klwt={kl_weight:.5f} lr={lr:.4f}",
                f"regwt={regression_loss_weight:.5f}",
            ]
            tqdm_data.set_postfix_str(" ".join(postfix))
            if i % self.config.tb_log_interval == 0:
                
                
                postfix = {
                #    "pct_match": pct_match,
                    "step": num_batches * epoch + i,
                    "epoch":epoch,
                    "i":i,
                    "kl_weight": kl_weight,
                    "lr": lr,
                    "kl_loss": kl_loss,
                    "recon_loss": recon_loss,
                    "regression_loss": regression_loss,
                    "regression_loss_weight": regression_loss_weight,
                    "loss": loss,
                    "mode": "Eval" if optimizer is None else "Train",
                }
                
                if i % self.config.save_frequency == 0:
                    pct_match = compute_reconstruction_trans(model, val_loader)
                    postfix.update({"pct_match":pct_match})
                    if pct_match > self.config.best_pct_match:
                        print('Saving model')
                        self.config.best_pct_match = pct_match
                        model = model.to("cpu")
                        torch.save(
                            model.state_dict(),
                            self.config.model_save[:-3] + ".pt",
                        )
                        torch.save(
                            optimizer.state_dict(),
                            self.config.opt_save[:-3] + ".pt",
                        )
                        torch.save(
                            postfix,
                            self.config.opt_save[:-3] + "_step.pt",
                        )

                        model = model.to(self.config.device)
                self.tb_log(postfix, num_batches * epoch + i)

        return postfix

    # def get_optim_params(self, model, regression_model):
    #     return (p for p in model.vae.parameters() if p.requires_grad)

    def get_optim_params(self, model, regression_model):
        params = [p for p in model.vae.parameters() if p.requires_grad]
        if regression_model:
            regression_model_params = [
                p for p in regression_model.parameters() if p.requires_grad
            ]
            params = params + regression_model_params
        return params

    def _train(
        self,
        model,
        regression_models,
        train_loader,
        val_loader=None,
        logger=None,
    ):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(
            self.get_optim_params(model, regression_models),
            lr=self.config.lr_start,
        )
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer, self.config)
        for i in range(self.config.start_epoch):
            lr_annealer.step()

        model.zero_grad()
        for epoch in range(self.config.start_epoch, n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(
                train_loader,
                desc="Training (epoch #{}) of #{}".format(epoch, n_epoch),
            )
            postfix = self._train_epoch(
                model,
                regression_models,
                epoch,
                tqdm_data,
                val_loader,
                kl_weight=kl_weight,
                optimizer=optimizer,
            )
            # if logger is not None:
            #     logger.append(postfix)
            #     self.tb_log(postfix, epoch)
            #     logger.save(self.config.log_file)

            # if val_loader is not None:
            #     tqdm_data = tqdm(
            #         val_loader, desc="Validation (epoch #{})".format(epoch)
            #     )
            #     postfix = self._train_epoch(
            #         model,
            #         regression_models,
            #         epoch,
            #         tqdm_data,
            #         kl_weight=kl_weight,
            #     )
            #     if logger is not None:
            #         logger.append(postfix)
            #         self.tb_log(postfix, epoch)
            #         logger.save(self.config.log_file)

            # if (self.config.model_save is not None) and (
            #     epoch % self.config.save_frequency == 0
            # ):
            #     model = model.to("cpu")
            #     torch.save(
            #         model.state_dict(),
            #         self.config.model_save[:-3] + ".pt",
            #     )
            #     torch.save(
            #         optimizer.state_dict(),
            #         self.config.opt_save[:-3] + ".pt",
            #     )
            #     model = model.to(device)

            # Epoch end
            lr_annealer.step()

    def tb_log(self, data, epoch):
        #print(data, epoch)
        for k in data:
            if not (k == "mode"):
                self.tb_writer.add_scalar(
                    "%s/%s" % (data["mode"], k), data[k], epoch
                )

    def fit(self, model, regression_model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(log_dir=self.config.tb_loc)

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (
            None
            if val_data is None
            else self.get_dataloader(model, val_data, shuffle=False)
        )
        #self.config.best_recon_loss = 100
        self.config.best_pct_match = 0

        # TODO : Pass a proper logger below
        self._train(
            model, regression_model, train_loader, val_loader, logger=None
        )
        self.tb_writer.close()
        return model

    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )

