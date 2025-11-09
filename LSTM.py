import torch
from torch import nn
import torch.autograd as autograd


# Detect GPU availability
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LSTMModel(nn.Module):
    """
    LSTM-based sentiment analysis model.
    Supports both random and pre-trained (GloVe) embeddings.
    """

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        embedding_matrix,
        hidden_dim,
        n_layers,
        input_len,
        pretrain=False
    ):
        super().__init__()

        self.output_size = output_size      # output (binary sentiment)
        self.n_layers = n_layers            # number of stacked LSTM layers
        self.hidden_dim = hidden_dim        # hidden state size
        self.input_len = input_len          # max sequence length

        # Embedding layer
        if pretrain:
            # Use pre-trained GloVe embeddings
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False
            )
        else:
            # Random initialization
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.init_weights()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Max pooling over sequence length
        self.pool = nn.MaxPool1d(self.input_len)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        """Initialize embedding weights for random initialization."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def _init_hidden(self, batch_size):
        """Initialize hidden and cell states."""
        hidden = autograd.Variable(
            torch.randn(self.n_layers, batch_size, self.hidden_dim)
        ).to(device)
        cell = autograd.Variable(
            torch.randn(self.n_layers, batch_size, self.hidden_dim)
        ).to(device)
        return hidden, cell

    def forward(self, x):
        
        batch_size = x.size(0)
        hidden_cell = self._init_hidden(batch_size)

        # 1. Convert word indices to embeddings
        embeds = self.embedding(x)

        # 2. Feed embeddings through LSTM
        lstm_out, _ = self.lstm(embeds, hidden_cell)

        # 3. Rearrange dimensions for pooling
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out = lstm_out.permute(0, 2, 1)  # -> [batch_size, hidden_dim, seq_len]

        # 4. Apply max pooling along sequence length dimension
        out = self.pool(lstm_out)  # -> [batch_size, hidden_dim, 1]

        # 5. Flatten the output
        out = out.view(out.size(0), -1)

        # 6. Apply dropout
        out = self.dropout(out)

        # 7. Feed through linear and sigmoid layers
        out = self.fc(out)
        out = self.sigmoid(out)

        # 8. Convert to 1D tensor
        out = out[:, 0]

        return out