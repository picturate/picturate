from picturate.imports import *
from picturate.nets.cycle_attngan import *

class CAttnGAN():

    def __init__(self, cfg, pretrained=False, cuda=False):
        self.cfg = cfg

        if cuda is False:
            self.cfg.CUDA = False

        self.cache_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../.weights')
        self.pretrained = pretrained
        
        # Load models
        self.load_models()

    
    def load_models(self):
        
        # Load word to index mapping
        word2idx_path = os.path.join(self.cache_directory, 'word2idx.pkl')
        self.word2idx = pickle.load(open(word2idx_path, 'rb'))

        # Load G_NET
        self.netG = G_NET(self.cfg)


        if self.pretrained is True:
            
            if not self.weight_exists('G_NET.pth'):
                pretrained_file_path = "https://drive.google.com/uc?export=download&id=1LHR_SsJo_YtihVunhluoWTIXGxFIsUSl"
                save_path = os.path.join(self.cache_directory, 'G_NET.pth')
                gdown.download(pretrained_file_path, save_path, quiet=False)

            G_NET_path = os.path.join(self.cache_directory, 'G_NET.pth')
            state_dict = torch.load(G_NET_path, map_location=lambda storage, loc: storage)
            self.netG.load_state_dict(state_dict)

            if self.cfg.CUDA:
                self.netG.cuda()
        
        self.netG.eval()

        # Load text encoder
        self.text_encoder = BERT_RNN_ENCODER(self.cfg.N_WORDS, cfg=self.cfg, nhidden=self.cfg.TEXT.EMBEDDING_DIM)

        if self.pretrained is True:

            if not self.weight_exists('BERT_RNN_ENCODER.pth'):
                pretrained_file_path = "https://drive.google.com/uc?export=download&id=1EqcAdwtbM5TU2qyQ7LSsDxlyLzJscKL6"
                save_path = os.path.join(self.cache_directory, 'G_NET.pth')
                gdown.download(pretrained_file_path, save_path, quiet=False)
    
            BERT_RNN_ENCODER_path = os.path.join(self.cache_directory, 'BERT_RNN_ENCODER.pth')
            state_dict = torch.load(BERT_RNN_ENCODER_path, map_location=lambda storage, loc: storage)
            self.text_encoder.load_state_dict(state_dict)

        self.text_encoder.eval()
        

    
    def get_tokens(self, sentence):
        data_dic = {}
        captions = []
        cap_lens = []

        sentence = sentence.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence.lower())
        
        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t)>0 and t in self.word2idx:
                rev.append(self.word2idx[t])
            
        captions.append(rev)
        cap_lens.append(len(rev))

        max_len = np.max(cap_lens)

        sorted_indices = np.argsort(cap_lens)[::-1]
        cap_lens = np.asarray(cap_lens)
        cap_lens = cap_lens[sorted_indices]
        cap_array = np.zeros((len(captions), max_len), dtype='int64')
        for i in range(len(captions)):
            idx = sorted_indices[i]
            cap = captions[idx]
            c_len = len(cap)
            cap_array[i, :c_len] = cap
        key = ' '.join(tokens)
        data_dic[key] = [cap_array, cap_lens, sorted_indices]

        return data_dic


    def generate_image(self, sentence, filename):
        data_dic = self.get_tokens(sentence)
        
        for key in data_dic:
            captions, cap_lens, sorted_indices = data_dic[key]
            batch_size = captions.shape[0]
            nz = self.cfg.GAN.Z_DIM
            
            captions = Variable(torch.from_numpy(captions))
            cap_lens = Variable(torch.from_numpy(cap_lens))

            noise = Variable(torch.FloatTensor(batch_size, nz))

            hidden = self.text_encoder.init_hidden(batch_size)

            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)

            noise.data.normal_(0,1)

            if self.cfg.CUDA:
                noise = noise.cuda()
                sent_emb = sent_emb.cuda()
                words_embs = words_embs.cuda()
                mask = mask.cuda()

            fake_imgs, attention_maps,_,_ = self.netG(noise, sent_emb, words_embs, mask)

            cap_lens_np = cap_lens.cpu().data.numpy()

            for j in range(batch_size):
                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0)*127.5
                    im = im.astype(np.uint8)

                    im = np.transpose(im, (1,2,0))

                    im = Image.fromarray(im)

                    im.save("{}_{}.png".format(filename, str(k)))

    def weight_exists(self, file_name):
        _path = os.path.join(self.cache_directory, file_name)
        return os.path.exists(_path)
