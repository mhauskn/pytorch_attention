import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)


    def forward(self, query, values, mask):
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden # TODO: Consider bridge
        decoder_outputs = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]
        query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(input_vocab_size, hidden_size)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden
