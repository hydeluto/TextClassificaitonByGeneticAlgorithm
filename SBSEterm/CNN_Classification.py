import common

import pandas as pd
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader

class TxtDataSet(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CNN(nn.Module):
    def __init__(self,
                 char_size: int,
                 embedding_dim: int,
                 n_filters: int,
                 filter_size,
                 output_dim: int):

        super().__init__()
        self.embedding = nn.Embedding(char_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_size]
        )
        self.fc = nn.Linear(len(filter_size) * n_filters, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        return self.fc(torch.cat(pooled, dim=1))

class SentimentAnalysisByCNN:
    def __init__(self,
                 character_count=30,
                 embedding_dimension=32,
                 num_filters=30,
                 num_filter_size=2,
                 output_dimension=2,
                 epoch=5,
                 batch_size=150,
                 learning_rate=1e-3,
                 weight_decay=1e-4):

        self.character_cnt = character_count
        self.embedding_dim = embedding_dimension
        self.n_filters = num_filters
        self.n_filter_size = [num_filter_size-1, num_filter_size, num_filter_size+1]
        self.output_dim = output_dimension
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.model = CNN(len(common.KOR_CHAR_LIST)+1, self.embedding_dim, self.n_filters, self.n_filter_size, self.output_dim)

    def preprocess_sentiment_data(self):
        char_to_idx = {char: i for i, char in enumerate(common.KOR_CHAR_LIST)}

        df_senti_train = pd.read_csv("ratings_train.csv")
        df_senti_test = pd.read_csv("ratings_test.csv")

        x_train = [[char_to_idx[c] if c in char_to_idx else len(char_to_idx) for c in s] for s in df_senti_train.text]
        y_train = np.array([int(i) for i in df_senti_train.label], np.int64)

        self.x_test = [[char_to_idx[c] if c in char_to_idx else len(char_to_idx) for c in s] for s in df_senti_test.text]
        self.y_test = np.array([int(i) for i in df_senti_test.label], np.int64)

        for ndx, d in enumerate(x_train):
            x_train[ndx] = np.pad(d, (0, self.character_cnt), 'constant', constant_values=0)[:self.character_cnt]
        for ndx, d in enumerate(self.x_test):
            self.x_test[ndx] = np.pad(d, (0, self.character_cnt), 'constant', constant_values=0)[:self.character_cnt]

        self.x_test = np.array(self.x_test, np.int64)
        x_train = torch.tensor(x_train).to(torch.int64)

        self.train_loader = DataLoader(dataset=TxtDataSet(x_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=TxtDataSet(self.x_test, self.y_test), batch_size=self.batch_size, shuffle=True)

    def train_eval_model(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        elapse_time, test_acc = 0, 0
        for epoch in range(self.epoch):
            start = time.time()
            train_loss, test_loss = 0, 0

            for x_i, y_i in self.train_loader:
                optimizer.zero_grad()

                output = self.model(x_i)
                loss = criterion(output, y_i)

                loss.backward()
                optimizer.step()

                train_loss += float(loss)

            with torch.no_grad():
                test_output = self.model(torch.tensor(self.x_test, dtype=torch.long))
                test_predict = torch.argmax(test_output, dim=1)
                test_correct = (test_predict == torch.tensor(self.y_test, dtype=torch.long))

                for x_j, y_j in self.test_loader:
                    self.model.eval()
                    te_output = self.model(x_j)
                    te_loss = criterion(te_output, y_j)
                    test_loss += float(te_loss.item())

                    test_acc = test_correct.sum().item() / len(self.x_test) * 100

            elapse_time = time.time() - start
            print('epoch: {}, loss : {:.2f}, acc: {:.2f}, time: {:.2f}'.format(
                epoch, train_loss / len(self.train_loader), test_acc, elapse_time))
            if elapse_time > 60 or test_acc < 75:
                elapse_time = math.inf
                break

        if(elapse_time == math.inf):
            return math.inf
        return elapse_time/test_acc

if __name__ == '__main__':
    sentiCNN = SentimentAnalysisByCNN()
    sentiCNN.preprocess_sentiment_data()
    sentiCNN.train_eval_model()