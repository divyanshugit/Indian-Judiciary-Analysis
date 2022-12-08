import torch
import evaluate

load_f1 = evaluate.load("f1")

def get_f1_score_from_logits(logits, labels): 
   probabilties = torch.sigmoid(logits.unsqueeze(-1))
   predictions = (probabilties > 0.5).long().squeeze()
   f1 = load_f1.compute(predictions=predictions, references=labels,average="weighted")['f1']
   # print(f1)
   return f1
