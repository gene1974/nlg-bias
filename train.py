import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import logger, save_model
from dataset import bert_data
from pytorchtools import EarlyStopping


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate(model, data_loader):
    all_label = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            logits = model(input_ids, attention_mask)['logits']
            pred = torch.argmax(logits, dim = 1)
            all_label.extend(labels.tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())
    acc = accuracy_score(all_label, all_pred)
    p = precision_score(all_label, all_pred, average = 'macro')
    r = recall_score(all_label, all_pred, average = 'macro')
    f1 = f1_score(all_label, all_pred, average = 'macro')
    # logger('[Eval] {} Acc: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(data_loader.__str__, acc, p, r, f1))
    return acc, p, r, f1


def train(model, epochs = 100, lr = 1e-6, patience = 5, early_stop = True):
    logger('Start training... Model: {}, lr = {}'.format(model.__class__.__name__, lr))
    optimizer = optim.Adam(model.parameters(), lr)
    early_stopping = EarlyStopping(patience = patience, verbose = False)
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_losses = []
        model.train()
        for _, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            optimizer.zero_grad()
            output = model(input_ids, attention_mask = attention_mask, labels = labels) # (n_batch, n_token, n_class)
            loss, pred_logits = output[:2]
            loss.backward()
            optimizer.step()

            # metrics
            train_losses.append(loss.item())
            predict = torch.argmax(pred_logits, dim = 1) # (n_batch)
            train_correct += torch.sum(predict == labels).item()
            train_total += len(labels)
        avg_train_loss = np.average(train_losses)

        acc, p, r, f1 = evaluate(model, valid_loader)
        # logger('[epoch {:d}] TLoss: {:.3f} VLoss: {:.3f} TAcc: {:.3f} VAcc: {:.3f}'.format(
        #     epoch + 1, avg_train_loss, avg_valid_loss, train_correct / train_total, valid_correct / valid_total))
        logger('[epoch {:d}] Loss: {:.3f}  TrainAcc: {:.3f} VAcc: {:.3f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
            epoch + 1, avg_train_loss, train_correct / train_total, acc, p, r, f1))
        
        # logger('Precision: {:.3f} Recall: {:.3f} F1: {:.3f}'.format(precision, recall, f1))
        # early_stopping(f1, model)
        early_stopping(-acc, model)
        if early_stop and early_stopping.early_stop:
            logger("Early stopping")
            break
    save_model(model, 'checkpoint/{}'.format(model.__class__.__name__))
    evaluate(model, test_loader)
    return model

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels = 3).to(device)
    train_loader, valid_loader, test_loader = bert_data()
    model = train(model, epochs = 100, lr = 1e-6, patience = 5, early_stop = True)
