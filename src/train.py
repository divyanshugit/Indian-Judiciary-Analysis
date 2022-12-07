import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import JusticeDataset
from model import Classifier


if __name__ == "__main__":

    classifier = Classifier("bert-base-uncased")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=classifier.model.parameters())

    # Initialize training set and loader.
    train_set = JusticeDataset(
            filename="/nlsasfs/home/ttbhashini/prathosh/divyanshu/PreCog/train_e.csv",
            maxlen=50,
            tokenizer=classifier.tokenizer,
        )
    val_set = JusticeDataset(
        filename="/nlsasfs/home/ttbhashini/prathosh/divyanshu/PreCog/val_e.csv",
            maxlen=50,
            tokenizer=classifier.tokenizer
    )

    # Initialize validation set and loader.
    train_loader = DataLoader(
        dataset=train_set, batch_size=16, num_workers=8
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=16, num_workers=8
    )

    # Initialize best accuracy.
    best_accuracy = 0
    for epoch in trange(5, desc="Epoch"):
        classifier.train(
            train_loader=train_loader, optimizer=optimizer, criterion=criterion
        )
        val_accuracy, val_loss = classifier.evaluate(
            val_loader=val_loader, criterion=criterion
        )
        
        print(
            f"Epoch {epoch} complete! Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}"
        )
        # Save classifier if validation accuracy imporoved.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(
                f"Best validation accuracy improved from {best_accuracy} to {val_accuracy}, saving classifier..."
            )
            # classifier.save()
