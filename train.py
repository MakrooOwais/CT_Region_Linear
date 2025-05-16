import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse

from datamodule import CT_Datamodule
from model import Classifier

def parse_args():
    parser = argparse.ArgumentParser(description='Train PPGL Genetic Cluster Classification Model')
    parser.add_argument('--data_dir', type=str, default='Dataset', 
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=5, 
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Maximum number of training epochs')
    parser.add_argument('--lr_dino', type=float, default=1e-5, 
                        help='Learning rate for backbone')
    parser.add_argument('--lr_class', type=float, default=1e-2, 
                        help='Learning rate for classification heads')
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help='Weight decay for optimizer')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU index to use')
    parser.add_argument('--folds', type=int, default=10, 
                        help='Number of cross-validation folds')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize data module
    data_module = CT_Datamodule(
        path=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_folds=args.folds
    )
    
    # Prepare data (load and split into folds)
    data_module.prepare_data()
    
    # Train each fold
    results = {}
    for fold_idx in range(args.folds):
        print(f"Training fold {fold_idx+1}/{args.folds}")
        
        # Set the current fold
        data_module.set_k(fold_idx)
        
        # Initialize model
        model = Classifier(
            lr_dino=args.lr_dino,
            lr_class=args.lr_class,
            weight_decay=args.weight_decay,
            k=fold_idx
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=os.path.join(args.output_dir, f'fold_{fold_idx}'),
            filename='model-{epoch:02d}-{val_acc:.4f}',
            save_top_k=3,
            mode='max',
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            patience=20,
            mode='max',
        )
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=os.path.join(args.output_dir, 'logs'),
            name=f'fold_{fold_idx}'
        )
        
        # Initialize trainer
        trainer = Trainer(
            accelerator="gpu",
            devices=[args.gpu],
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            log_every_n_steps=1,
            deterministic=True
        )
        
        # Train model
        trainer.fit(model, data_module)
        
        # Test model
        results[fold_idx] = trainer.test(model, data_module)[0]
        
        # Print current fold results
        print(f"Fold {fold_idx+1} results:")
        for metric, value in results[fold_idx].items():
            print(f"  {metric}: {value:.4f}")
    
    # Calculate average results across folds
    avg_results = {}
    for metric in results[0].keys():
        avg_results[metric] = sum(fold[metric] for fold in results.values()) / args.folds
    
    # Print final results
    print("\nAverage results across all folds:")
    for metric, value in avg_results.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
