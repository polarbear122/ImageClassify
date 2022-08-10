import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler


def test_kfold(dataset):
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42


    # Usage Example:
    num_epochs = 10    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    for epoch in range(num_epochs):
        # Train:
        for batch_index, (faces, labels) in enumerate(train_loader):
            pass
