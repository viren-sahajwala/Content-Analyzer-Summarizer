from __future__ import unicode_literals
from flask import Blueprint, render_template, request, redirect, url_for
import re
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from wordcloud import WordCloud, STOPWORDS


from .models import *
import spacy
nlp = spacy.load('en_core_web_sm')

views = Blueprint('views', __name__)

@views.route('/', methods = ['GET', 'POST'])
def home(): 
    return render_template("home.html")

@views.route('/lnk', methods=['GET', 'POST'])
def link():
    return render_template("link_summ.html")

@views.route('/about')
def about():
    return render_template("about.html")

@views.route('/txt_summarize',methods=['GET','POST'])
def txt_summarize():
    if request.method == 'POST':
        rawtext = request.form['tt_txt']
        rawtext = re.sub("\[[0-9]+\]", '', rawtext)
        raw_count = word_count(rawtext)
        
        # Spacy Summarizer
        spacy_final_summary = spacy_summ(rawtext)
        sp = reading_time(spacy_final_summary, 5)
        sp_count = word_count(spacy_final_summary)

        # LexRank Summarizer
        lex_final_summary = lex_summ(rawtext)
        lx = reading_time(lex_final_summary, 5)
        lx_count = word_count(lex_final_summary)
        
        # Gensim Summarization
        gensim_final_summary = gen_summ(rawtext)
        gs = reading_time(gensim_final_summary, 5)
        gs_count = word_count(gensim_final_summary)

        # NLTK Summarization
        nltk_final_summary = nltk_summ(rawtext)
        nn = reading_time(nltk_final_summary, 5)
        nn_count = word_count(nltk_final_summary)

        # generating wordcloud
        stopwords = set(STOPWORDS)
        fig = Figure()
        axis = fig.add_subplot(1,1,1)
        wordcloud = WordCloud(background_color="white", stopwords = stopwords).generate(spacy_final_summary)
        axis.imshow(wordcloud)
        axis.axis("off")
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
    
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        return render_template("summary.html", 
        rawtext = rawtext, raw_count = raw_count, 
        spacy_final_summary = spacy_final_summary, sp = sp, sp_count = sp_count, 
        lex_final_summary = lex_final_summary, lx=lx, lx_count = lx_count,
        gensim_final_summary = gensim_final_summary, gs = gs, gs_count = gs_count,
        nltk_final_summary = nltk_final_summary, nn = nn, nn_count = nn_count, 
        wordcloud = pngImageB64String)

    else: 
        return redirect('/')



@views.route('/lnk_summarize',methods=['GET','POST'])
def lnk_summarize():
    if request.method == 'POST':
        raw_url = request.form['tt_lnk']
        rawtext = get_text(raw_url)
        rawtext = re.sub("\[[0-9]+\]", '', rawtext)
        raw_count = word_count(rawtext)
        
        # Spacy Summarizer
        spacy_final_summary = spacy_summ(rawtext)
        sp = reading_time(spacy_final_summary, 5)
        sp_count = word_count(spacy_final_summary)

        # LexRank Summarizer
        lex_final_summary = lex_summ(rawtext)
        lx = reading_time(lex_final_summary, 5)
        lx_count = word_count(lex_final_summary)

        # Gensim Summarization
        gensim_final_summary = gen_summ(rawtext)
        gs = reading_time(gensim_final_summary, 5)
        gs_count = word_count(gensim_final_summary)

        # NLTK Summarization
        nltk_final_summary = nltk_summ(rawtext)
        nn = reading_time(nltk_final_summary, 5)
        nn_count = word_count(nltk_final_summary)

        # generating wordcloud
        stopwords = set(STOPWORDS)
        fig = Figure()
        axis = fig.add_subplot(1,1,1)
        wordcloud = WordCloud(background_color="white", stopwords = stopwords).generate(spacy_final_summary)
        axis.imshow(wordcloud)
        axis.axis("off")
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
    
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        return render_template("summary.html", 
        rawtext = rawtext, raw_url= raw_url, raw_count = raw_count,
        spacy_final_summary = spacy_final_summary, sp = sp, sp_count = sp_count,
        lex_final_summary = lex_final_summary, lx=lx, lx_count = lx_count,
        gensim_final_summary = gensim_final_summary, gs = gs, gs_count = gs_count,
        nltk_final_summary = nltk_final_summary, nn = nn, nn_count = nn_count,
        wordcloud = pngImageB64String)
    
    else:
        return redirect('/lnk')

