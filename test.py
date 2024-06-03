from transformers import BertForSequenceClassification, BertConfig


config = BertConfig.from_pretrained("models/bert-base-uncased", num_labels=2)

model = BertForSequenceClassification.from_pretrained("models/bert-base-uncased", config=config)


