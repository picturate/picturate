from picturate.imports import *
from picturate.nets.cycle_attngan import *

class CAttnGAN():

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../.weights')
        
        # Load models
        self.load_models()

    
    def load_models(self):
        
        # Load word to index mapping
        word2idx_path = os.path.join(self.cache_directory, 'word2idx.pkl')
        self.word2idx = pickle.load(open(word2idx_path, 'rb'))

        # Load 

    
    def encode_text(self, sentence):
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


    def generate_image(self, sentence):
        data_dic = self.encode_text(sentence)
        print(data_dic)
    
    def test(self):
        print("Works")