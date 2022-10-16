import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tqdm import trange
from metrics import b_metrics
from tokenizer import preprocess

class Trainer:
    def __init__(self):
        self.text = None
        self.labels = None
        self.dataset = 'SMSSpamCollection'
        self.tokenizer = None
        self.token_id = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(
                            'bert-base-uncased',
                            do_lower_case = True
                            )


    def convert(self):
        df = pd.DataFrame({'label':int(), 'text':str()}, index=[])
        with open(self.dataset,'r') as f:
            for line in f.readlines():
                split = line.split('\t')
                df = df.append({'labels': 1 if split[0] == 'spam' else 0,
                                'text': split[1]},
                                ignore_index=True)
                self.text = df.text.values
                self.labels = df.labels.values

        for sample in self.text:
            encoding_dict = preprocess(sample, self.tokenizer)
            self.token_id.append(encoding_dict['input_ids'])
            self.attention_masks.append(encoding_dict['attention_mask'])

    def split_dataset(self):
        train_idx, val_idx = train_test_split(
                                    np.arange(len(self.labels)),
                                    test_size = 0.2,
                                    shuffle = True,
                                    stratify = self.labels
                                )
        train_set = TensorDataset(self.token_id[train_idx],
                                self.attention_masks[train_idx],
                                self.labels[train_idx])

        val_set = TensorDataset(self.token_id[val_idx],
                                self.attention_masks[val_idx],
                                self.labels[val_idx])

        self.train_dataloader = DataLoader(
                                        train_set,
                                        sampler = RandomSampler(train_set),
                                        batch_size=16
                                    )

        self.validation_dataloader = DataLoader(
                                        val_set,
                                        sampler = RandomSampler(val_set),
                                        batch_size = 16
                                    )
        
    def train(self):
        self.model = BertForSequenceClassification.from_pretrained(
                                    'bert-base-uncased',
                                    num_labels = 2,
                                    output_attentions = False,
                                    output_hidden_states = False,
                                )

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=5e-5,
                                      eps=1e-08)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = 2
        for _ in trange(epochs, desc='Epochs'):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                # forward pass
                train_output = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask = b_input_mask,
                                     labels=b_labels)

                # backward pass
                train_output.loss.backward()
                optimizer.step()
                # update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            # validation
            self.model.eval()

            val_accuracy = []
            val_precision = []
            val_recall = []
            val_specificity = []

            for batch in self.validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    # forward pass
                    eval_output = self.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask)
                    logits = eval_output.logits.detach().cpu().numpy()
                    labels_ids = b_labels.to('cpu').numpy()

                    #validation metrics
                    b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, labels_ids)
                    val_accuracy.append(b_accuracy)

                    # update precision only when (tp+fp)!=0; ignore nan
                    if b_precision != 'nan': val_precision.append(b_precision)
                    # update recall only when (tp + fn) !=0; ignore nan
                    if b_recall != 'nan': val_recall.append(b_recall)
                    # update specificity only when (tn + fp) !=0; ignore nan
                    if b_specificity != 'nan': val_specificity.append(b_specificity)

                print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
                print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
                print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
                print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
                print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

    def predict(self, sentence):
        test_ids = []
        test_attention_mask = []
        encoding = preprocess(sentence, self.tokenizer)

        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        with torch.no_grad():
            output = self.model(test_ids.to('cpu'),
                                token_type_ids=None,
                                attention_mask=test_attention_mask.to('cpu'))

        prediction = 'spam' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Ham'      

        print('Input Sentence: ', sentence)
        print('Predicted Class: ', prediction)

obj = Trainer()
obj.convert()
obj.split_dataset()
obj.train()
obj.predict('free for a day')