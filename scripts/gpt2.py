# script for calculate suprisal and uncertainties

import pickle
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = TFGPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id = tokenizer.eos_token_id, return_dict = True)

def get_features(df):
    setups = df['set-up'].tolist()
    punchlines = df['punchline'].tolist()

    surprisals = []
    uncertainties = []
    for setup, punchline in tqdm(zip(setups, punchlines), total = len(setups)):
        setup_ids = tokenizer.encode(setup, return_tensors = 'tf')
        punchline_ids = tokenizer.encode(punchline, return_tensors = 'tf')

        all_ids = tokenizer.encode(setup + ' ' + punchline, return_tensors = 'tf')

        outputs = model(all_ids[:,:-1])
        logits = outputs.logits[0,-(punchline_ids.shape[1]):,:]
        scores = tf.nn.softmax(logits, 1)
        log_scores = logits - tf.expand_dims(tf.reduce_logsumexp(logits, 1), 1)

        uncertainty = -tf.reduce_mean(tf.reduce_sum(scores * log_scores, 1))
        uncertainties.append(uncertainty.numpy())

        labels = tf.one_hot(all_ids[0,-(punchline_ids.shape[1]):], logits.shape[-1])
        if labels.shape[0] == log_scores.shape[0]:
            loss = -tf.reduce_mean(tf.reduce_sum(labels * log_scores, 1))
            surprisals.append(loss.numpy())
        else:
            surprisals.append(-1.0)  # Ignore examples with surprisal value == -1.0

    return surprisals, uncertainties

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    surprisals, uncertainties = get_features(df)
    pd.DataFrame({
        'surprisal': surprisals,
        'uncertainty': uncertainties
    }).to_csv('surprisal_and_uncertainty.csv', index = False)
