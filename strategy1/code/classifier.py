import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from dataloader import MultiAvtDataLoader 
from dataloader import collate_fn
torch.set_default_tensor_type(torch.DoubleTensor)
from torchmetrics.classification import BinaryF1Score
import torchmetrics
from focal_loss.focal_loss import FocalLoss
import pandas as pd
class MultiAvtDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def setup(self, stage=None):
        self.train_data = MultiAvtDataLoader(self.args, train_csv1=self.args.train_file, video_feats_paths = self.args.train_video_location,  split = "train")
        self.val_data = MultiAvtDataLoader(self.args, train_csv1=self.args.val_file, video_feats_paths = self.args.val_video_location,  split = "val")
        N = len(self.train_data)
 
        print("Number of training samples ==> {}".format(len(self.train_data)))
        print("Number of validation samples ==> {}".format(len(self.val_data)))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, collate_fn=collate_fn, num_workers=self.args.num_workers)


class MultiAvtClassifier(pl.LightningModule):
    def __init__(self, in_dim, do_p, lr):
        super().__init__()
        self.target_ids = []
        self.post_ids = []
        # self.layers = nn.Sequential(*[
        #         nn.Linear(in_dim, 512),
        #         # nn.Dropout(p = do_p),
        #         nn.ReLU(),
            
        #         nn.Linear(512, 256),
        #         nn.Dropout(p = do_p),
        #         nn.ReLU(),
            
        #         nn.Linear(256, 256),
        #         nn.Dropout(p = do_p),
        #         nn.ReLU(),

        #         nn.Linear(256, 64),
        #         nn.Dropout(p = do_p),
        #         nn.ReLU(),

        #         nn.Linear(64, 1),
        #         nn.Sigmoid()
        # ])
        
        self.layers = nn.Sequential(*[
                nn.Linear(in_dim, 512),
                nn.Dropout(p = do_p),
                nn.ReLU(),
            
                nn.BatchNorm1d(num_features=512),

                nn.Linear(512, 256),
                nn.Dropout(p = do_p),
                nn.ReLU(),
            
                nn.BatchNorm1d(num_features=256),

                nn.Linear(256, 256),
                nn.Dropout(p = do_p),
                nn.ReLU(),

                nn.BatchNorm1d(num_features=256), 

                nn.Linear(256, 64),
                nn.Dropout(p = do_p),
                nn.ReLU(),

                nn.BatchNorm1d(num_features=64),

                nn.Linear(64, 1),
                nn.Sigmoid()
        ])
        

        self.lr = lr
        self.nn_loss = torch.nn.BCELoss()
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        
        self.train_f1 = BinaryF1Score(task="binary")
        self.val_f1 = BinaryF1Score(task="binary")
        
        self.save_hyperparameters()

    def forward(self, x, x_text=None):
        output = self.layers(x)
        return output

    def get_loss(self, net_output, gt_labels):
        ce=self.nn_loss(net_output,gt_labels)
        return ce

    def training_step(self, train_batch, batch_idx):
        x = train_batch['data']
        y_labels= train_batch["target"]
        ids=train_batch["id"]
        logits = self.forward(x)
        y_labels = y_labels.unsqueeze(1)
        y_labels =y_labels.type('torch.cuda.DoubleTensor')
        loss = self.get_loss(logits, gt_labels= y_labels)
        self.train_accuracy(logits, y_labels)
        self.train_f1(logits, y_labels)
        self.log('train_acc', self.train_accuracy.compute(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('tr_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss}
    
        
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data']
        y_labels= val_batch["target"]
        ids=val_batch["id"]
        logits = self.forward(x)
        y_labels = y_labels.unsqueeze(1)
        y_labels =y_labels.type('torch.cuda.DoubleTensor')
        loss = self.get_loss(logits,gt_labels= y_labels )
        self.val_accuracy(logits, y_labels)
        self.val_f1(logits, y_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', self.val_accuracy.compute(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return {'ids':ids, 'preds': logits, 'target' : y_labels, 'loss': loss, 'val_acc': self.val_accuracy.compute(), 'val_f1': self.val_f1.compute()}

    def validation_epoch_end(self, outputs):
        ids = torch.cat([x['ids'] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        targets = torch.cat([x['target'] for x in outputs]).detach().cpu().numpy()
        post_ids = [] 
        preds_list = []
        targets_list = []

        for id in ids:
            post_ids.append(id)
        for pred in preds:
            preds_list.append(pred)
        for target in targets:
            targets_list.append(target)

        df = pd.DataFrame({'id': post_ids, 'preds': preds_list, 'target': targets_list})
        df.to_csv("best_results.csv", index=False)
        return {'preds': preds, 'target': targets, 'ids': ids}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input', type=int, default=512)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--save_pred', type=str, default=None)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3) 
        return [optimizer], [scheduler]