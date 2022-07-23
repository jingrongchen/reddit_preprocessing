import os
import re
import nltk
import spacy
import random
import neuralcoref
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import tensorflow as tf

nltk.download('wordnet')
nltk.download('stopwords')
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

transition_phrases = ['thus', 'for example', 'for instance', 'namely', 'to illustrate',
                      'in other words', 'in particular', 'specifically', 'such as',
                      'on the contrary', 'contrarily', 'notwithstanding', 'but', 'however',
                      'nevertheless', 'in spite of', 'in contrast', 'yet', 'on one hand',
                      'on the other hand', 'rather', 'or', 'nor', 'conversely', 'at the same time',
                      'while this may be true', 'and', 'in addition to', 'furthermore',
                      'moreover', 'besides', 'than', 'too', 'also', 'both-and', 'another',
                      'equally important', 'second', 'etc.', 'again', 'further', 'last',
                      'finally', 'not only-but also', 'as well as', 'in the second place',
                      'next', 'likewise', 'similarly', 'in fact', 'as a result', 'consequently',
                      'in the same way', 'for example', 'for instance', 'however', 'thus',
                      'therefore', 'otherwise', 'after that', 'afterward', 'then', 'next',
                      'last', 'at last', 'at length', 'at first', 'formerly', 'another', 'finally',
                      'meanwhile', 'at the same time', 'afterwards', 'subsequently',
                      'in the meantime', 'eventually', 'concurrently', 'simultaneously', 'although',
                      'at least', 'still', 'even though', 'granted that', 'while it may be true',
                      'in spite of', 'of course', 'similarly', 'likewise', 'in like fashion',
                      'in like manner', 'analogous to', 'above all', 'indeed', 'of course',
                      'certainly', 'surely', 'in fact', 'really', 'in truth', 'again', 'besides',
                      'also', 'furthermore', 'in addition', 'specifically', 'especially',
                      'in particular', 'to explain', 'to list', 'to enumerate', 'in detail',
                      'namely', 'including', 'for example', 'for instance', 'to illustrate',
                      'thus', 'in other words', 'as an illustration', 'in particular', 'so that',
                      'with the result that', 'consequently', 'hence', 'accordingly', 'for this reason',
                      'therefore', 'because', 'due to', 'as a result', 'in other words', 'then',
                      'therefore', 'finally', 'consequently', 'thus', 'in conclusion', 'as a result',
                      'accordingly', 'for this purpose', 'to this end', 'with this in mind',
                      'with this purpose in mind', 'therefore']

def transition_start(first_sent, dialog_turn):
    if dialog_turn == 1:
        for phrase in transition_phrases:
            if first_sent.lower().startswith(phrase):
                return True
        return False
    else:
        return False                



def smmry(text, doc, sent_count, dialog_turn):

    # some preprocessing to omit text within brackets and replace u with you. 
    
    # text = re.sub("[\(\[].*?[\)\]]", "", text)
    # text = text.replace(' u ', ' you ')

    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    # doc = nlp(text)

    fdist = {}
    word_arr = nltk.word_tokenize(formatted_text.lower())

    # preparing a frequency dictionary without considering stop words
    
    for word in word_arr:
        if not word in stop_words:
            word = wnl.lemmatize(word)
            if word not in fdist.keys():
                    fdist[word] = 1
            else:
                    fdist[word] += 1

    sent_arr = nltk.sent_tokenize(text)
    sent_score_arr = []
    summary_arr = []

    sent_arr_coref_resolved = nltk.sent_tokenize(doc._.coref_resolved)

    # compute scores for each sentence

    for sent in sent_arr:
        score = 0
        token_arr = nltk.word_tokenize(sent.lower())
        for word in token_arr:
            word = wnl.lemmatize(word)
            if word in fdist.keys():
                score += fdist[word]

        sent_score_arr.append(score/len(token_arr))

    sent_score_arr = np.array(sent_score_arr)

    all_ind_arr = sent_score_arr.argsort()[-len(sent_score_arr):][::-1]

    ind_arr_unsorted = sent_score_arr.argsort()[-sent_count:][::-1]

    ind_arr = np.sort(ind_arr_unsorted) 

    summary = ''
    changed_first = False

    if len(ind_arr) > 0:

        try:

            ind = ind_arr[0]
            first_sent = sent_arr[ind]

            while (first_sent != sent_arr_coref_resolved[ind] or transition_start(first_sent, dialog_turn)):
                changed_first = True
                for index in all_ind_arr:
                    if index < ind:
                        ind = index
                        break
                first_sent = sent_arr[ind]
                if ind == 0:
                    break
            summary = summary + first_sent + ' '     
            
            if (changed_first):
                first_ind = ind
                sent_score_modified = sent_score_arr[first_ind+1:]
                ind_arr_unsorted = sent_score_modified.argsort()[-(sent_count-1):][::-1]
                ind_arr_next = np.sort(ind_arr_unsorted) 
                
                for i in range(0, len(ind_arr_next)):
                    ind = (first_ind+1) + ind_arr_next[i]
                    if i == len(ind_arr_next) - 1:
                        summary = summary + sent_arr[ind]
                    else:
                        summary = summary + sent_arr[ind] + ' '
            
            else:
                for i in range(1, len(ind_arr)):
                    ind = ind_arr[i]
                    if i == len(ind_arr) - 1:
                        summary = summary + sent_arr[ind]
                    else:
                        summary = summary + sent_arr[ind] + ' '

            return summary

        except Exception as e:

            print("EXCEPTION occured")
            return text

    else:
        print(text)
        print(sent_arr)
        print("EXCEPTION occured: length of sentence array is not > 0")
        return text


def preprocess_raw(text):
    # Check if text is a str
    if not isinstance(text, str):
        return None

    # Replace HTML escape chars
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&amp;', '&')
    text = text.replace('#x200B;', ' ')
    text = text.replace('nbsp;', ' ')

    # Remove brackets
    b_pattern = re.compile(r'(\([^\(\)]*\))|(\[[^\[\]]*\])')
    while b_pattern.search(text):
        text = re.sub(r'(\([^\(\)]*\))|(\[[^\[\]]*\])', '', text)

    # Remove redundant spaces (including breaklines)
    text = ' '.join(text.split())

    # Check if text is empty
    if not text:
        return None

    text_lower = text.lower()

    # Check if text is [deleted] or [removed]
    if text_lower == '[deleted]' or text_lower == '[removed]':
        return None

    # Check if text contains URL
    url_pattern = re.compile(r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    if url_pattern.search(text_lower):
        return None

    # Check if text contains 'r/<subreddit>' or 'u/<username>'
    r_pattern = re.compile(r'(^| )\/?r\/[^ ]*')
    if r_pattern.search(text_lower):
        return None
    u_pattern = re.compile(r'(^| )\/?u\/[^ ]*')
    if u_pattern.search(text_lower):
        return None

    # Check if text contains 'reddit'
    if 'reddit' in text_lower:
        return None

    # Check the percentage of alphabetical letters
    num_alphas = 0
    for ch in text:
        if ch.isalpha():
            num_alphas += 1
    if num_alphas / len(text) < 0.7:
        return None

    doc = nlp(text)

    # Check the number of tokens
    if len(doc) < 2:
        return None

    return {'text': text, 'doc': doc}        





def preprocess_summary(text):
    # Check if text is a str
    if not isinstance(text, str):
        return None

    # Remove redundant spaces (including breaklines)
    text = ' '.join(text.split())

    # Check if text is empty
    if not text:
        return None

    # Check the percentage of alphabetical letters
    num_alphas = 0
    for ch in text:
        if ch.isalpha():
            num_alphas += 1
    if num_alphas / len(text) < 0.7:
        return None

    doc = nlp(text)

    # Check the number of tokens
    if len(doc) < 2:
        return None

    return {'text': text, 'doc': doc}


def extract_root(text, sent):
#     # Check if the last character is a question mark
#     if sent[-1].text == '?':
#         return None

    if sent.root.pos_ == 'VERB':
#         # Check the first token
#         if sent[0].lemma_.lower() in q_words:
#             return None
        return sent.root.lemma_

    return None


def summarize(preprocessed_text, dialog_turn):
    if preprocessed_text is None:
        return None

    text = preprocessed_text['text']
    doc = preprocessed_text['doc']

    summarized = 0
    sents = [sent for sent in doc.sents]
    if len(sents) > 1:
        summarized = 1
        summary = smmry(text, doc, 1, dialog_turn)
        preprocessed_summary = preprocess_summary(summary)
        if preprocessed_summary is None:
            return None
        summarized_text = preprocessed_summary['text']
        summarized_doc = preprocessed_summary['doc']
        summarized_sents = [sent for sent in summarized_doc.sents]
        if len(summarized_sents) != 1:
            return None
    elif len(sents) == 1:
        summarized_text = text
        summarized_doc = doc
        summarized_sents = sents
    else:
        return None

    if dialog_turn > 1:
        return {'text': summarized_text, 'summarized': summarized, 'length': len(summarized_sents[0])}

    root = extract_root(summarized_text, summarized_sents[0])
    if root is not None:
        return {'text': summarized_text, 'summarized': summarized, 'root': root, 'length': len(summarized_sents[0])}
    else:
        return None



date_suffices = ['20210501_20211231', '20210101_20210430', '20200101_20201231',
                 '20190101_20191231', '20180101_20181231', '20170101_20171231',
                 '20160101_20161231']

    

for date_suffix in date_suffices:
    submission_filtered_df = pd.read_csv('../data/filtered_q/submission/casual_conv_submissions_{}.csv'.format(date_suffix))
    comment_filtered_df = pd.read_csv('../data/filtered_q/comment/casual_conv_comments_{}.csv'.format(date_suffix))
    comment_filtered_df=comment_filtered_df.drop_duplicates()

    final_cols = ['src_id', 'src_type', 'src_summarized', 'src_from', 'src_text', 'src_root', 'src_length',
                  'com_id', 'com_summarized', 'com_text', 'com_length','com_score']
    final_dict = {col: [] for col in final_cols}
    
    comment_filtered_parent_df=comment_filtered_df[comment_filtered_df['parent_id'].str.startswith('t3')]
    comment_filtered_parent_ids = comment_filtered_parent_df['parent_id'].apply(lambda x:x.split('_')[1]).tolist()
    
    submission_filtered_df = submission_filtered_df[submission_filtered_df['id'].isin(comment_filtered_parent_ids)]
    print(submission_filtered_df.shape)

    
    print(date_suffix,'Start assemble submission with comment')
    for i in tqdm(range(submission_filtered_df.shape[0])):       
        sub_id = submission_filtered_df.iloc[i]['id']
        sub_summarized = submission_filtered_df.iloc[i]['summarized']
        sub_from = submission_filtered_df.iloc[i]['from']
        sub_text = submission_filtered_df.iloc[i]['text']
        sub_root = submission_filtered_df.iloc[i]['root']
        sub_length = submission_filtered_df.iloc[i]['length']
        comment_filtered_df_sub = comment_filtered_df[comment_filtered_df['parent_id'] == 't3_' + sub_id]
        for j in range(comment_filtered_df_sub.shape[0]):
            final_dict['src_id'].append(sub_id)
            final_dict['src_type'].append('sub')
            final_dict['src_summarized'].append(sub_summarized)
            final_dict['src_from'].append(sub_from)
            final_dict['src_text'].append(sub_text)
            final_dict['src_root'].append(sub_root)
            final_dict['src_length'].append(sub_length)
            final_dict['com_id'].append(comment_filtered_df_sub.iloc[j]['id'])
            final_dict['com_summarized'].append(comment_filtered_df_sub.iloc[j]['summarized'])
            final_dict['com_text'].append(comment_filtered_df_sub.iloc[j]['text'])
            final_dict['com_length'].append(comment_filtered_df_sub.iloc[j]['length'])
            final_dict['com_score'].append(comment_filtered_df_sub.iloc[j]['score'])
    
    
    
    coment_coment_df=comment_filtered_df[comment_filtered_df['parent_id'].str.startswith('t1')]
    coment_coment_ids=coment_coment_df['parent_id'].apply(lambda x:x.split('_')[1]).tolist()    
    comment_have_kid_df = comment_filtered_df[comment_filtered_df['id'].isin(coment_coment_ids)]
    
    print(date_suffix,'Start assemble comment with comment')
    for k in tqdm(range(comment_have_kid_df.shape[0])):       
        src_id = comment_have_kid_df.iloc[k]['id']
        src_summarized = comment_have_kid_df.iloc[k]['summarized']
        src_text = comment_have_kid_df.iloc[k]['text']
        src_length = comment_have_kid_df.iloc[k]['length']
        
        comment_comment_df = comment_filtered_df[comment_filtered_df['parent_id'] == 't1_' + src_id]
        for m in range(comment_comment_df.shape[0]):
            final_dict['src_id'].append(src_id)
            final_dict['src_type'].append('com')
            final_dict['src_summarized'].append(src_summarized)
            final_dict['src_from'].append('comment')
            final_dict['src_text'].append(src_text)
            final_dict['src_root'].append(src_root)
            final_dict['src_length'].append(src_length)
            
            final_dict['com_id'].append(comment_comment_df.iloc[m]['id'])
            final_dict['com_summarized'].append(comment_comment_df.iloc[m]['summarized'])
            final_dict['com_text'].append(comment_comment_df.iloc[m]['text'])
            final_dict['com_length'].append(comment_comment_df.iloc[m]['length'])
            final_dict['com_score'].append(comment_comment_df.iloc[m]['score'])
    
    final_df = pd.DataFrame(final_dict)
    print("final df shape:",final_df.shape)
    final_df.to_csv('../data/matched_q/casual_conv_{}.csv'.format(date_suffix), index = False)




