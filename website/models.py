# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests

# Fetch Text From Url
def get_text(url):
  res = requests.get(url)
  soup = BeautifulSoup(res.content, 'html.parser')
  fetched_text = ''
  fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
  return fetched_text

#import numpy as np
# estimating the average time to read the data
def reading_time(text_list, word_length):
  WPM = 200
  total_words = 0
  for current_text in text_list:
    total_words += len(current_text)/word_length
  return round(total_words/WPM, 2)

# creating wordcloud
from matplotlib.figure import Figure
from wordcloud import WordCloud

def word_cloud(text):
    fig = Figure(figsize = (12,12))
    axis = fig.add_subplot(1, 1, 1)
    wordcloud = WordCloud(background_color="white").generate(text)
    axis.imshow(wordcloud)
    axis.axis("off")
    return fig

# counting the number of words
def word_count(str):
    counts = len(str.split())
    return counts

# NLP Pkgs

# Spacy and related libraries 
import spacy 
nlp = spacy.load('en_core_web_sm')

# Pkgs for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq for Finding the Top N Sentences
from heapq import nlargest


# Spacy Summarization
def spacy_summ(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}  
    for word in docx:  
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Sentence Scores
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]


    summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    return summary








# LexRank packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk


# LexRank Summarization
def lex_summ(raw_docx):
    summary = ''
    my_parser = PlaintextParser.from_string(raw_docx,Tokenizer('english'))
    lex_rank_summarizer = LexRankSummarizer()
    lexrank_summary = lex_rank_summarizer(my_parser.document, sentences_count=8)
    for sentence in lexrank_summary:
        sentence = str(sentence)
        summary += sentence
    return summary 





# Gensim Packages
import gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

# Gensim Summarization 
def gen_summ(raw_docx):
    summary = summarize(raw_docx, word_count = 150)
    return summary








# NLTK Packages
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  

# NLTK Summarization
def nltk_summ(raw_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(raw_text):
        if word not in stopWords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())
    
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
    sentence_list = nltk.sent_tokenize(raw_text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    
    summary_sentences = heapq.nlargest(6, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary



















'''
# importing BART
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# Loading the model and tokenizer for bart-large-cnn
tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# Bart Summarization
def bart_summ(raw_docx):
    inputs = tokenizer.batch_encode_plus([raw_docx],return_tensors='pt',truncation=True)
    summary_ids = model.generate(inputs['input_ids'], early_stopping=False)
    bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return bart_summary


# Bert Summarization
from summarizer import Summarizer,TransformerSummarizer
bert_model = Summarizer()
def bert_summ(raw_docx):
    bert_summary = ''.join(bert_model(raw_docx, max_length=200))
    return bert_summary

# GPT-2 Summarization   
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
def gpt2_summ(raw_docx):
  gpt_summary = ''.join(GPT2_model(raw_docx, max_length=200))
  return gpt_summary
'''
