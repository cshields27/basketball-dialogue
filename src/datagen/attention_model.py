from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax
from tensorflow.compat.v1.keras.layers import CuDNNGRU
import keras
import keras.utils
import tensorflow as tf
from keras import metrics

'''
This was created using Professor McMillan's guide here:
https://github.com/mcmillco/funcom/blob/master/models/attendgru.py
'''

class AttentionGRUModel:
    def __init__(self, config):
        self.config = config
        self.cvocabsize = config['c_vocabsize']
        self.qvocabsize = config['q_vocabsize']
        self.avocabsize = config['a_vocabsize']
        self.clen = config['clen']
        self.qlen = config['qlen']
        self.alen = config['alen']
        
        self.embdims = 100
        self.recdims = 100
        self.rnndims = 100

        self.config['num_input'] = 3
        self.config['num_output'] = 1

    def create_model(self):            
        q_input = Input(shape=(self.qlen,))
        a_input = Input(shape=(self.alen,))
        c_input = Input(shape=(self.clen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.qvocabsize, mask_zero=False, weights=self.config['weights'])(q_input)
        se = Embedding(output_dim=self.embdims, input_dim=self.cvocabsize, mask_zero=False)(c_input)

        se_enc = GRU(self.rnndims, return_state=True, return_sequences=True)
        #se_enc = CuDNNGRU(self.rnndims, return_state=True, return_sequences=True) # for GPU ?
        seout, state_sml = se_enc(se)

        enc = GRU(self.rnndims, return_state=True, return_sequences=True)
        #enc = CuDNNGRU(self.rnndims, return_state=True, return_sequences=True) # for GPU ?
        encout, state_h = enc(ee, initial_state=state_sml)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.avocabsize, mask_zero=False)(a_input)
        dec = GRU(self.rnndims, return_sequences=True)
        # dec = CuDNNGRU(self.rnndims, return_sequences=True) # for GPU ?
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)

        c_attn = dot([decout, seout], axes=[2, 2])
        c_attn = Activation('softmax')(c_attn)

        context = dot([attn, encout], axes=[2, 1])
        c_context = dot([c_attn, seout], axes=[2, 1])

        context = concatenate([context, decout, c_context])

        out = TimeDistributed(Dense(self.rnndims, activation="relu"))(context)

        out = Flatten()(out)
        out = Dense(self.avocabsize, activation="softmax")(out)

        model = Model(inputs=[q_input, a_input, c_input], outputs=out)

        '''if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)'''
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model