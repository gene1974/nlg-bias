import logging
def logger(content):
    logging.getLogger('matplotlib.font_manager').disabled = True
    log_format = '[%(asctime)s] %(message)s'
    date_format = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level = logging.DEBUG, format = log_format, datefmt = date_format)
    logging.info(content)

import time
import torch
def save_model(model, path):
    ts = time.strftime('%m%d%H%M', time.localtime())
    torch.save(model.state_dict(), '{}.{}'.format(path, ts))
    print('Save model to', '{}.{}'.format(path, ts))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate(model, data_loader):
    all_label = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels, texts = batch
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

