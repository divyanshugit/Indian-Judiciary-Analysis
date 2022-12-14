import torch
import torch.nn as nn
from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from utils import get_f1_score_from_logits

class DistilBertForClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # DistilBERT.
        self.distilbert = DistilBertModel(config)
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = outputs.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_reps)
        return logits


class Classifier:
    def __init__(self,model_name):
        self.config = AutoConfig.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForClassification.from_pretrained(model_name,num_labels=3)
        # requires_grad(self.model, True)

        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.model.to(self.device)


    # Evaluates classifier.
    def evaluate(self, val_loader, criterion):
        # Set model to evaluation mode.
        self.model.eval()
        batch_f1_score_summation, loss, num_batches = 0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(
                val_loader, desc="Evaluating"
            ):
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_f1_score_summation += get_f1_score_from_logits(logits, labels)
                loss += criterion(logits.squeeze(-1), labels.squeeze(-1).float()).item()
                num_batches += 1
        f1_score = batch_f1_score_summation / num_batches
        loss = loss / num_batches
        return f1_score, loss

    # Trains classifier for one epoch.
    def train(self, train_loader, optimizer, criterion):
        self.model.train()
        for input_ids, attention_mask, labels in tqdm(
            iterable=train_loader, desc="Training"
        ):
            # Reset gradient
            optimizer.zero_grad()
            input_ids, attention_mask, labels = (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                labels.to(self.device),
            )
            
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(input=logits.squeeze(-1), target=labels.squeeze(-1).float())
            loss.backward()
            optimizer.step()

        return loss

    # Saves fine tuned classifer model.
    def save(self):
        self.model.save_pretrained(save_directory=f"models/")
        self.config.save_pretrained(save_directory=f"models/")
        self.tokenizer.save_pretrained(save_directory=f"models/")
