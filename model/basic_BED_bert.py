from model.config import config
from keras import Input
from keras_bert import load_trained_model_from_checkpoint,Tokenizer
class event_detection:
    def __init__(self):
        self.token_dict = self.collect_token_dict()

    def collect_token_dict(self):
        token_dict = {}
        with open(config.vocab_dir, mode='r', encoding='utf-8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict

    def input_layer(self):
        self.sent_a = Input(shape=(None,),name='sent_input_a')
        self.sent_b = Input(shape=(None,),name='sent_input_b')

    def bert_layer(self):
        bert_model = load_trained_model_from_checkpoint(config_file=config.config_dir,checkpoint_file=config.model_dir)

        for l in bert_model.layers:
            l.tranable = True

        bert_embed = bert_model(inputs=[self.sent_a,self.sent_b])
        # sent_embedding = Lambda(lambda x: x[:,0,:])(bert_embed)
        # word_embedding = Lambda(lambda x: x[:,1:-1,:])(bert_embed)

        print(bert_embed[:,1:-1,:].shape)

    def compile(self):
        self.input_layer()
        self.bert_layer()

if __name__ == '__main__':
    ed = event_detection()
    ed.compile()