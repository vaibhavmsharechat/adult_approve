import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from classifier import MultiAvtDatamodule
torch.set_default_tensor_type(torch.DoubleTensor)
from classifier import MultiAvtClassifier 

def main(args):
    pl.utilities.seed.seed_everything(seed=43)    
    data_module = MultiAvtDatamodule(args)
    model = MultiAvtClassifier(args.input, args.dropout, args.lr)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1', verbose=True, save_top_k=1, mode='max')
    early_stopping_callback = EarlyStopping(monitor="val_f1", mode="max", min_delta = 0.01, patience=2 )
    trainer = pl.Trainer(gpus=[0], max_epochs=100, callbacks= [checkpoint_callback, early_stopping_callback],num_sanity_val_steps=0,default_root_dir="/home/vaibhavmishra/adult_approve/strategy1/code")
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run', type=str, default='00')
    parser = MultiAvtClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--train_file', type=str, default="/home/vaibhavmishra/adult_approve/strategy1/dataset/dataset/vaibhav_AA_discard_safe_all_combined_train_all_info_with_feats_step7.csv")
    parser.add_argument('--val_file', type=str, default="/home/vaibhavmishra/adult_approve/strategy1/dataset/dataset/vaibhav_AA_discard_safe_all_combined_val_all_info_with_feats_step7.csv")
    parser.add_argument('--batch_size', type=int, default=2056)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_video_location', type=str, default="/home/vaibhavmishra/adult_approve/strategy1/dataset/dataset/features/train_npy/clip/")
    parser.add_argument('--val_video_location', type=str, default="/home/vaibhavmishra/adult_approve/strategy1/dataset/dataset/features/val_npy/clip/")
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    print(args)
    if args.phase == "train":
        main(args)