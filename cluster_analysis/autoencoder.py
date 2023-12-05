import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CharLevelAutoencoder(nn.Module):
    """
    A simple character-level autoencoder with RNN layers for encoding and decoding text data.
    This autoencoder is designed to work with sequences of characters and includes an additional
    task to predict metadata based on the encoded representation.
    """
    def __init__(self, seq_length, char_vocab_size, metadata_dim, embedding_dim=50, rnn_dim=256, latent_dim=64):
        """
        Initialize the autoencoder model with the given dimensions and layers.
        Parameters:
        ----------
        seq_length : int
            The length of input sequence.
        char_vocab_size : int
            The number of unique characters in the vocabulary.
        embedding_dim : int
            The dimensionality of character embeddings.
        rnn_dim : int
            The number of features in the hidden state of the RNNs.
        metadata_dim : int
            The number of metadata features to predict.
        latent_dim : int
            The dimensionality of the latent space (encoded representations).
        """
        super(CharLevelAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim)
        self.encoder_rnn = nn.RNN(embedding_dim, rnn_dim, batch_first=True)
        self.decoder_rnn = nn.RNN(latent_dim, rnn_dim, batch_first=True)
        self.to_latent = nn.Linear(rnn_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, rnn_dim)
        self.output_layer = nn.Linear(rnn_dim, char_vocab_size)
        self.metadata_layer = nn.Linear(latent_dim, metadata_dim)

    def encode(self, x):
        """
        Encode a sequence of characters into a latent space representation.
        Parameters:
        ----------
        x : torch.Tensor
            The input tensor containing character indices.
        Returns:
        -------
        torch.Tensor
            The encoded latent representation.
        """
        embeds = self.embedding(x)
        _, hidden = self.encoder_rnn(embeds)
        latent = self.to_latent(hidden[-1])
        return latent

    def decode(self, latent):
        """
        Decode a latent space representation back into a sequence of characters.
        Parameters:
        ----------
        latent : torch.Tensor
            The latent space representation to decode.
        """
        seq_length = self.seq_length
        latent = latent.unsqueeze(0).repeat(seq_length, 1, 1)
        rnn_out, _ = self.decoder_rnn(latent)
        out = self.output_layer(rnn_out)
        return out.transpose(0, 1).transpose(1, 2)


    def forward(self, x):
        """
        The forward pass of the autoencoder.
        Parameters:
        ----------
        x : torch.Tensor
            The input tensor containing character indices.
        """
        latent= self.encode(x)
        reconstructed = self.decode(latent)
        
        metadata_pred = self.metadata_layer(latent)
        return reconstructed, metadata_pred


class CharVectorizer:
    """
    A training class for the CharLevelAutoencoder to handle training loops, loss calculation, and optimization.
    """
    def __init__(self, model, data_loader, optimizer, criterion=nn.CrossEntropyLoss(), metadata_criterion=nn.BCEWithLogitsLoss() , device='cpu'):
        """
        Initialize the trainer with the required components.
        Parameters:
        ----------
        model : CharLevelAutoencoder
            The autoencoder model to be trained.
        data_loader : DataLoader
            The DataLoader that provides batches of data for training.
        optimizer : torch.optim.Optimizer
            The optimizer to use for updating model weights.
        criterion : torch.nn.modules.loss._Loss
            The loss function for the reconstruction task.
        metadata_criterion : torch.nn.modules.loss._Loss
            The loss function for the metadata prediction task.
        device : str, optional
            The device to use for training ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metadata_criterion = metadata_criterion
        self.device = device

    def train_epoch(self, metadata_loss_weight):
        """
        Run a single epoch of training.
        Parameters:
        -----------
        meta_loss_weight : float
            weight of metadata loss.
        """
        self.model.train()
        total_loss = 0

        for batch in self.data_loader:
            inputs, metadata = batch
            inputs, metadata = inputs.to(self.device), metadata.to(self.device)
            metadata = metadata.float()

            self.optimizer.zero_grad()
            reconstructed, metadata_pred = self.model(inputs)

            reconstruction_loss = self.criterion(reconstructed, inputs)
            metadata_loss = self.metadata_criterion(metadata_pred, metadata)

            loss = reconstruction_loss + metadata_loss_weight * metadata_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.data_loader)
        print(f'Average Loss: {average_loss:.4f}')

    def train(self, epochs, metadata_loss_weight=0.5):
        """
        Run the training process for a given number of epochs.
        Parameters:
        ----------
        epochs : int
            The number of epochs to train the model for.
        """
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            self.train_epoch(metadata_loss_weight)
    
    def encode_data(self, data):
        """
        Encode data using the autoencoder's encoder.
        Paramters:
        ----------
            data (torch.Tensor): Data to encode.
        Returns:
            torch.Tensor: Encoded data.
        """
        self.model.eval()
        with torch.no_grad():
            encoded_data = self.model.encode(data.to(self.device))
        return encoded_data


class TextDataset(Dataset):
    def __init__(self, dataframe, meta_lst, padding_char=' '):
        """
        Initialize the TextDataset.
        Parameters:
        ----------
            dataframe (pandas.DataFrame): The data in a pandas DataFrame.
            meta_lst (list): List of metadata column names in the dataframe.
            padding_char (str, optional): Character used for padding text. Defaults to a space.
        """
        self.data_frame = dataframe
        self.meta_lst = meta_lst
        self.max_length = self.calculate_max_length()
        self.char_to_index = self.create_char_to_index_map(padding_char)

    def create_char_to_index_map(self, padding_char):
        """
        Create a mapping from characters to indices.
        Parameters:
        ----------
            padding_char (str): Character used for padding.
        Returns:
            dict: Mapping of characters to indices.
        """
        char_to_index = {chr(i): i - 97 for i in range(97, 123)}  # Map a-z to 0-25
        char_to_index[padding_char] = 26  # Map padding character to 26
        return char_to_index

    def calculate_max_length(self):
        """
        Calculate the maximum length of text in the dataset.
        Parameters:
        ----------
            int: The maximum length of text.
        """
        return self.data_frame['phrase'].str.len().max()

    def __len__(self):
        """
        Get the number of items in the dataset.
        Parameters:
        ----------
            int: Number of items.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.
        Parameters:
        ----------
            idx (int): Index of the item.
        Returns:
            tuple: (text_tensor, metadata_tensor) - Tensors of text and metadata.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data_frame.iloc[idx]['phrase']
        text_tensor = self.process_text(text)

        metadata = self.data_frame.iloc[idx][self.meta_lst].values.astype(int)
        metadata_tensor = torch.tensor(metadata, dtype=torch.long)

        return text_tensor, metadata_tensor

    def process_text(self, text):
        """
        Process the text into a tensor of character indices.
        Parameters:
        ----------
            text (str): The text to process.
        Returns:
            torch.Tensor: Tensor of character indices.
        """
        text_indices = [self.char_to_index.get(char, 26) for char in text.lower()[:self.max_length]] 
        text_indices += [self.char_to_index[' ']] * (self.max_length - len(text_indices))
        return torch.tensor(text_indices, dtype=torch.long)





    

    
