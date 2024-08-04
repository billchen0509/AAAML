# lstm.py

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        '''
        :param embedding_matrix: np.array
        '''
        super(LSTM, self).__init__()
        # number of words = number of rows in the embedding matrix
        num_words = embedding_matrix.shape[0]

        # dimension of the word embeddings is num of columns in the embedding matrix
        embed_dim = embedding_matrix.shape[1]

        # we define an embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # embedding matrix is used to initialize the weights of the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        # we donnot want to train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # a simple bidirectional LSTM with hidden size of 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )

        # output layer which is a linear layer
        # we have only one output
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        # pass the input through the embedding layer
        x = self.embedding(x)

        # move embedding output to the lstm
        x, _ = self.lstm(x)

        # apply mean and max pooling on the lstm output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction = 256
        # avg_pool = 256, max_pool = 256
        out = torch.cat((avg_pool, max_pool), 1)

        # pass through the output layer and return the output
        out = self.out(out)

        return out
    