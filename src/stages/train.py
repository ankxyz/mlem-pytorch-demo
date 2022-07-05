"""
Trains classification model
"""

import argparse
from mlem.api import save
from pathlib import Path
import time
import torch
from tqdm import tqdm
from typing import Text

from src.utils.config import load_config
from src.utils.datasets import ImageFolderWithPaths
from src.utils.loggers import get_logger
from src.utils.train import get_loss_fn, Network
from src.utils.transforms import get_transforms


def train(config_path: Text) -> None:
    """Trains gesture classification model.
    Args:
        config_path(Text): path to config
    """

    config = load_config(config_path)
    logger = get_logger('TRAIN', log_level=config.base.log_level)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Device: {device}')

    logger.info('Load datasets')
    data_transform = get_transforms()
    
    # Create dataset objects from folder/images
    raw_data_dir = Path(config.base.raw_data_dir)
    train_dataset = ImageFolderWithPaths(
        root=raw_data_dir / 'train',
        transform=data_transform
    )
    
    # Create data loader
    logger.info('Create Data Loaders')
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.base.batch_size,
        shuffle=True,
        num_workers=config.base.num_workers
    )

    logger.info('Setup model')

    classes_number = len(train_dataset.classes)
    logger.info(f'Classes number = {classes_number}')
    
    model = Network(out_features=classes_number)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.step_size,
        gamma=config.train.gamma
    )

    logger.info('Fit model')
    epochs = config.train.epochs
    loss_fn = get_loss_fn(device)

    start_time = time.time()
    sample_data = torch.Tensor(0)

    for epoch in tqdm(range(epochs)):

        logger.info(f'Epoch {epoch}/{epochs}')  
        logger.info('-' * 10)
        logger.info('Train')
        
        epoch_loss = 0
        epoch_acc = 0
        for i, (img, label, _) in tqdm(enumerate(trainloader)):
            img = img.to(device)
            sample_data = img
            label = label.to(device)
            
            predict = model(img)
            loss = loss_fn(predict, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            correct_prediction = torch.argmax(predict, 1) == label
            correct_prediction = correct_prediction.sum()
            epoch_acc += correct_prediction / img.shape[0]
            
        epoch_loss = epoch_loss / len(trainloader)
        epoch_acc = epoch_acc / len(trainloader)

        print('Epoch : {}/{},   loss : {:.5f},    acc : {:.5f}'.format(epoch+1, epochs, epoch_loss, epoch_acc))
        
        if epoch_acc > 0.99 and epoch_loss < 0.1 :
            print('early stop')
            break

        time_elapsed = time.time() - start_time
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60
        logger.info(f'Training complete in {minutes:.0f}m {seconds:.0f}s') 

        scheduler.step()

    save(
        obj=model,
        path=f'clf_model',
        sample_data=sample_data,
        description=f'PyTorch classification model'
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(args.config)
