import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from pyvis.network import Network
import subprocess
from ast import literal_eval


import tensorflow as tf
colber_model = tf.keras.models.load_model("../Colbert/colbert-trained/")
colber_model.summary()


from transformers import BertTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
MODEL_TYPE = 'bert-base-uncased'
colber_tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)


def score(sentence):
    tokenize_input = colber_tokenizer.encode(sentence
    tensor_input = torch.tensor([tokenize_input])
    loss=colber_model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())



def return_id(str1, str2, truncation_strategy, length):

    inputs = colber_tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = length - len(input_ids)
    padding_id = colber_tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, colber_tokenizer):
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])
    
    for _, row in tqdm(df[columns].iterrows()):
        i = 0
        
        # sent
        sentences = sent_tokenize(row.combined)
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        # full row
        ids_q, masks_q, segments_q = return_id(row.combined, None, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    print(model_input[0].shape)
    return model_input



import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import re
import math

import torch
import sys
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from torch.nn import functional as F
from scipy.special import softmax
import string



# cuda=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load pre-trained model (weights)
with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')



def com_svalue(setup,punchline):
    if setup[-1] in punc:
        sentence=setup+' '+punchline
    else:
        sentence=setup+'. '+punchline


    #         print(sentence)
    tokenize_all = tokenizer.encode(sentence)
    #         print(len(tokenize_all))
    #         print('tokenize_all:',tokenize_all)

    setup_len=len(tokenizer.encode(setup))
    #         pun_ids=tokenizer.encode(punchline)
    #         print('setup',len(tokenizer.encode(setup)))
    #         print('pun',len(pun_ids))

    tensor_input = torch.tensor([tokenize_all])

    outputs = model(tensor_input)
    logit=outputs.logits[:,setup_len::,:]
    



    tensor1 = logit.detach().numpy()
    pre=softmax(tensor1)

    #         print(pre.shape)

    


    pun_len=len(tokenize_all)-setup_len
    #         print('pun_len',pun_len)
    s_value=0
    for i in range(0,pun_len):
        j=tokenize_all[setup_len+i]

        # log_scores = pre[0][i][j] - tf.expand_dims(tf.reduce_logsumexp(pre[0][i][j], 1), 1)
        
        if pre[0][i][j] > 0:
            s_value=s_value+math.log(pre[0][i][j])
            # s_value=s_value+log_scores
    s_value=-s_value/len(tokenize_all)
    return s_value

def filter_per(x):
    num_com=len(x)
    if num_com>=2:     
        df=x[(x['com_text'].str.contains('|'.join(searchfor)))|((x['com_text'].str.contains(regrex_pattern,regex= True, na=False)))]
        num_fun=len(df)
        per=num_fun/num_com
        if  per>0.4:
            return df    
    else:
        return None

def concate_fun(x): 
    src_id=x.split('_')[1]
    if x.startswith('t3'):
        df=org_submission_df[org_submission_df['id']==src_id].head(1)
        if len(df)>0:
            return df['text'].item()
       
    else:
        df=org_comment_df[org_comment_df['id']==src_id].head(1)
        if len(df)>0:
            return df['text'].item() 

MAX_SENTENCE_LENGTH = 20
MAX_SENTENCES = 5
MAX_LENGTH = 100
punc = string.punctuation
searchfor=['<3','xd',';D',':D',
           '🙃','😂','🤣', 
           'funny','hilarious','kidding','laughing so hard','going nuts','so funny','laughing','laughing hard','jokes',
           'take upvote','good joke','nice one','good one','joke always comments','old joke'
           ]

emoticon=['<3','xd',';D']
emoji=['🙃','😂','🤣']

# regrex_pattern=r'\b(a*ha+h[ha]*|o?l+o+l+[ol]*|e*he+h[he]*|lmao+)\b'
regrex_pattern='a*ha+h[ha]*|o?l+o+l+[ol]*|e*he+h[he]*|lmao+|:D+'

# match 'haha','hahah','hahaha','hahahaha','ahaha'
#         'he'......
#        lolololol
#        lmao, lmaoooo



date_suffices = [
'20170101_20171231', '20180101_20181231',
                 '20200101_20201231', '20210101_20211231']

for date_suffix in date_suffices:

    org_comment_df = pd.read_csv('../data/askreddit/filtered_q/comment/askReddit_comment_{}.csv'.format(date_suffix))
    org_submission_df = pd.read_csv('../data/askreddit/filtered_q/submission/askReddit_submission_{}.csv'.format(date_suffix))
    #bigger than 50 percentage of coment are hahaha and bigger than tow comments
    
    ##########################################################################################################generating filtered_df
    first_comment_df=org_comment_df[(org_comment_df['text'].str.contains('|'.join(searchfor)))|((org_comment_df['text'].str.contains(regrex_pattern,regex= True, na=False)))]
    #t1_ids是回复他的回复里有好笑的回答的指向
    t1_ids=first_comment_df['parent_id'].apply(lambda x:x.split('_')[1]).tolist()
    filtered_ids=[x for x in t1_ids]
    final_cols = ['src_id', 'src_parent','src_type', 'src_summarized', 'src_from', 'src_text', 'src_root', 'src_length',
                  'com_id', 'com_summarized', 'com_text', 'com_length','com_score']
    final_dict = {col: [] for col in final_cols}
    comment_have_kid_df = org_comment_df[org_comment_df['id'].isin(t1_ids)]
    print(date_suffix,'Start assemble comment(have fun indicater) with all of its comment')
    for k in tqdm(range(comment_have_kid_df.shape[0])):       
        src_id = comment_have_kid_df.iloc[k]['id']
        src_parent=comment_have_kid_df.iloc[k]['parent_id']
        src_summarized = comment_have_kid_df.iloc[k]['summarized']
        src_text = comment_have_kid_df.iloc[k]['text']
        src_length = comment_have_kid_df.iloc[k]['length']
        src_score   = comment_have_kid_df.iloc[k]['score']

        comment_comment_df = org_comment_df[org_comment_df['parent_id'] == 't1_' + src_id]
        for m in range(comment_comment_df.shape[0]):
            final_dict['src_id'].append(src_id)
            final_dict['src_parent'].append(src_parent)
            final_dict['src_type'].append('com')
            final_dict['src_summarized'].append(src_summarized)
            final_dict['src_from'].append('comment')
            final_dict['src_text'].append(src_text)
            final_dict['src_root'].append('none')
            final_dict['src_length'].append(src_length)
            
            final_dict['com_id'].append(comment_comment_df.iloc[m]['id'])
            final_dict['com_summarized'].append(comment_comment_df.iloc[m]['summarized'])
            final_dict['com_text'].append(comment_comment_df.iloc[m]['text'])
            final_dict['com_length'].append(comment_comment_df.iloc[m]['length'])
            final_dict['com_score'].append(src_score)

    filtered_df = pd.DataFrame(final_dict)
    
    print("filtered_df df shape:",filtered_df.shape)
    filtered_df=filtered_df.drop_duplicates()
    filtered_df=filtered_df.drop(['src_type', 'src_summarized', 'src_from','src_root', 'src_length','com_summarized','com_length'], axis=1)

    ########################################################################################################################################
    
    final_df=filtered_df.groupby("src_id").apply(filter_per)
    final_df=filtered_df

    if len(final_df)>0:
        final_df.insert(0,'reply_text','')
        final_df['reply_text']=final_df.progress_apply(lambda x:concate_fun(x['src_parent']),axis=1)
    #     org_submission_df[org_submission_df['id']==x['src_parent']]['text']
        final_df=final_df.dropna(subset=['reply_text'])
        #['reply_text', 'src_id', 'src_parent', 'src_text', 'com_id', 'com_text','com_score']
        final_df['combined'] = final_df.apply(lambda x:(x['reply_text']+' '+x['src_text']) if x['reply_text'][-1] in punc else (x['reply_text']+'. '+x['src_text']),axis=1)
        final_df.insert(0,'humor','')

        input_categories=['combined']
        print('doing gpt2')
        final_df['gpt2_score']=final_df.progress_apply(lambda x:com_svalue(x['src_text'],x['com_text']),axis=1)
        print('number of data to process',len(final_df))
        test_inputs = compute_input_arrays(final_df, input_categories, colber_tokenizer)
        final_df['colbert_score']=colber_model.predict(test_inputs)


        print(date_suffix,'done')
        # final_df=final_df.drop(['reply_text', 'src_parent', 'src_text', 'com_id', 'com_text'], axis=1)
        final_df=final_df.drop(['src_parent', 'src_id','com_id', 'com_text',], axis=1)
        final_df=final_df.drop_duplicates(subset=['combined'], keep='last')
        final_df.to_csv('../data/askreddit/final/askreddit_final_{}.csv'.format(date_suffix), index = False)
