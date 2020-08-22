from picturate.imports import *


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
    )


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
    )


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(
        self,
        ntoken,
        cfg,
        ninput=300,
        drop_prob=0.5,
        nhidden=128,
        nlayers=1,
        bidirectional=True,
    ):
        super(RNN_ENCODER, self).__init__()
        self.cfg = cfg
        self.n_steps = self.cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = self.cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == "LSTM":
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                Variable(
                    weight.new(
                        self.nlayers * self.num_directions, bsz, self.nhidden
                    ).zero_()
                ),
                Variable(
                    weight.new(
                        self.nlayers * self.num_directions, bsz, self.nhidden
                    ).zero_()
                ),
            )
        else:
            return Variable(
                weight.new(
                    self.nlayers * self.num_directions, bsz, self.nhidden
                ).zero_()
            )

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == "LSTM":
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class BERT_RNN_ENCODER(RNN_ENCODER):
    def define_module(self):
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.bert_linear = nn.Linear(768, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == "LSTM":
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.bert_linear.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb, _ = self.encoder(captions, output_all_encoded_layers=False)
        emb = self.bert_linear(emb)
        emb = self.drop(emb)
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == "LSTM":
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef, cfg):
        super(CNN_ENCODER, self).__init__()
        self.cfg = cfg
        if self.cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():  # freeze inception model
            param.requires_grad = False
        print("Load pretrained model from ", url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = Upsample(size=(299, 299), mode="bilinear")(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)  # nef

        if features is not None:
            features = self.emb_features(features)  # 17 x 17 x nef
        return features, cnn_code


# ############## Image2text Encoder-Decoder #######
class CNN_ENCODER_RNN_DECODER(CNN_ENCODER):
    def __init__(
        self,
        emb_size,
        hidden_size,
        vocab_size,
        nlayers=1,
        bidirectional=True,
        rec_unit="LSTM",
        dropout=0.5,
    ):
        """
        Based on https://github.com/komiya-m/MirrorGAN/blob/master/model.py
        :param emb_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        self.dropout = dropout
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        __rec_units = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }
        assert rec_unit in __rec_units, "Specified recurrent unit is not available"

        super().__init__(emb_size)

        self.hidden_linear = nn.Linear(emb_size, hidden_size)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.rnn = __rec_units[rec_unit](
            emb_size,
            hidden_size,
            num_layers=self.nlayers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.out = nn.Linear(self.num_directions * hidden_size, vocab_size)

    def forward(self, x, captions):
        # (bs x 17 x 17 x nef), (bs x nef)
        features, cnn_code = super().forward(x)
        # (bs x nef)
        cnn_hidden = self.hidden_linear(cnn_code)
        # (bs x hidden_size)

        #  (num_layers * num_directions, batch, hidden_size)
        h_0 = cnn_hidden.unsqueeze(0).repeat(self.nlayers * self.num_directions, 1, 1)
        c_0 = torch.zeros(h_0.shape).to(h_0.device)

        # bs x T x vocab_size
        text_embeddings = self.encoder(captions)
        # bs x T x nef
        output, (hn, cn) = self.rnn(text_embeddings, (h_0, c_0))
        # bs, T, hidden_size
        logits = self.out(output)
        # bs, T, vocab_size

        return features, cnn_code, logits


class BERT_CNN_ENCODER_RNN_DECODER(CNN_ENCODER):
    def __init__(
        self,
        emb_size,
        hidden_size,
        vocab_size,
        nlayers=1,
        bidirectional=True,
        rec_unit="LSTM",
        dropout=0.5,
    ):
        """
        Based on https://github.com/komiya-m/MirrorGAN/blob/master/model.py
        :param emb_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        self.dropout = dropout
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        __rec_units = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }
        assert rec_unit in __rec_units, "Specified recurrent unit is not available"

        super().__init__(emb_size)

        self.hidden_linear = nn.Linear(emb_size, hidden_size)
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.bert_linear = nn.Linear(768, emb_size)
        self.rnn = __rec_units[rec_unit](
            emb_size,
            hidden_size,
            num_layers=self.nlayers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.out = nn.Linear(self.num_directions * hidden_size, vocab_size)

    def forward(self, x, captions):
        # (bs x 17 x 17 x nef), (bs x nef)
        features, cnn_code = super().forward(x)
        # (bs x nef)
        cnn_hidden = self.hidden_linear(cnn_code)
        # (bs x hidden_size)

        #  (num_layers * num_directions, batch, hidden_size)
        h_0 = cnn_hidden.unsqueeze(0).repeat(self.nlayers * self.num_directions, 1, 1)
        c_0 = torch.zeros(h_0.shape).to(h_0.device)

        # bs x T x vocab_size
        # get last layer of bert encoder
        text_embeddings, _ = self.encoder(captions, output_all_encoded_layers=False)
        # bs x T x 768
        text_embeddings = self.bert_linear(text_embeddings)
        # bs x T x emb_size
        output, (hn, cn) = self.rnn(text_embeddings, (h_0, c_0))
        # bs, T, hidden_size
        logits = self.out(output)
        # bs, T, vocab_size

        return features, cnn_code, logits
