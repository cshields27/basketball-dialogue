from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf
from keras import metrics

# This was created using Professor McMillan's guide here:
# https://github.com/mcmillco/funcom/blob/master/models/attendgru.py

class AstAttentionGRUModel:
    def __init__(self, config):
        self.config = config
        self.c_vocabsize = config['c_vocabsize']
        self.q_vocabsize = config['q_vocabsize']
        self.a_vocabsize = config['a_vocabsize']
        self.clen = config['clen']
        self.qlen = config['qlen']
        self.alen = config['alen']
        
        self.embdims = 100
        self.recdims = 256

        self.config['num_input'] = 2
        self.config['num_output'] = 1

    def create_model(self):        
        a_input = Input(shape=(self.alen,))
        q_input = Input(shape=(self.qlen,))
        c_input = Input(shape=(self.clen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.avocabsize, mask_zero=False)(a_input)
        se = Embedding(output_dim=self.embdims, input_dim=self.cvocabsize, mask_zero=False)(c_input)

        se_enc = LSTM(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        enc = LSTM(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee, initial_state=state_sml)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.qvocabsize, mask_zero=False)(q_input)
        dec = LSTM(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, encout], axes=[2, 1])

        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)
        ast_context = dot([ast_attn, seout], axes=[2, 1])

        context = concatenate([context, decout, ast_context])

        out = TimeDistributed(Dense(self.recdims, activation="relu"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[a_input, q_input, c_input], outputs=out)

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model