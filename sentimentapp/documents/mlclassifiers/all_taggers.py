
# coding: utf-8

# In[18]:

import ner
import re
from textblob import Blobber, TextBlob
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import *
from textblob.sentiments import NaiveBayesAnalyzer
import os
#from pycorenlp import StanfordCoreNLP

tb = Blobber(analyzer=NaiveBayesAnalyzer())
#nlp = StanfordCoreNLP('http://localhost:9000')
annotator = SentimentIntensityAnalyzer()
tagger = ner.SocketNER(host='localhost', port=9191, output_format='slashTags')


def entity_highlighter(sentiment_output):
    # Replace NER Tags with 
    clean_html = re.sub("([\w\.\-]*)/ORGANIZATION","<font color = '0101DF''><b>\\1</b></font>", sentiment_output)
    clean_html = re.sub("([\w\.\-]*)/LOCATION","<font color = '5F076B''><b>\\1</b></font>", clean_html)
    clean_html = re.sub("([\w\.\-]*)/PERSON","<font color = 'DCCA0C''><b>\\1</b></font>", clean_html)
    return clean_html


def bracket_corrector(text):
    text = re.sub("-LRB-\s", "(", text)
    text = re.sub("\s-RRB-", ")", text)
    return text


def sentiment_highlighter(sentence, polarity, threshold_satisfy):
    if polarity > 0.0 and threshold_satisfy:
        sentence = "<font color = '60AB1E'>"+sentence+"</font>"
    elif polarity < 0.0 and  threshold_satisfy:
        sentence = "<font color = 'F23756'>"+sentence+"</font>"

    # No highlighting if neutral
    return sentence + ' '


# In[22]:

def sentiment_classifier(tag_output, classifer, threshold):
    sentence_list = sent_tokenize(tag_output)
    html_text = ''

    if classifer == 'TextBlob_Lexicon':
        for sentence in sentence_list:
            sent_blob = TextBlob(sentence)
            threshold_satisfy = np.abs(sent_blob.sentiment[0])>float(threshold)
            html_text += sentiment_highlighter(sentence, sent_blob.sentiment[0], threshold_satisfy)

    elif classifer == 'NLTK_VADER':
        for sentence in sentence_list:
            #vader_score = 0.0
            #vader_pol = annotator.polarity_scores(sentence)
            vader_score = annotator.polarity_scores(sentence)['compound']
            threshold_satisfy = np.abs(vader_score)>float(threshold)
            # if vader_pol['neg']>vader_pol['pos']:
            #     vader_score = -1.0
            # elif vader_pol['pos']>vader_pol['neg']:
            #     vader_score = 1.0
            html_text += sentiment_highlighter(sentence, vader_score, threshold_satisfy)

    elif classifer == 'TextBlob_Naive_Bayes':
        for sentence in sentence_list:
            #sent_blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
            sentence = re.sub(r'[^\x00-\x7F\n]+',' ', sentence.encode('utf-8'))
            nb_score = 0.0
            nb_pol = tb(sentence).sentiment
            if nb_pol[0] == 'neg':
                nb_score = -1.0
            elif nb_pol[0] == 'pos':
                nb_score = 1.0
            threshold_satisfy = np.abs(nb_pol[1]-nb_pol[2])>float(threshold)
            html_text += sentiment_highlighter(sentence, np.sign(nb_score), threshold_satisfy)
        
    elif classifer == 'CoreNLP':
        # The portion below gives only the class

        # for sentence in sentence_list:
        #     sentence = re.sub(r'[^\x00-\x7F\n]+',' ', sentence.encode('utf-8'))
        #     res = nlp.annotate(str(sentence),
        #                        properties={
        #             'annotators': 'sentiment',
        #             'outputFormat': 'json'
        #             })
        #     sentiment_class = 0.0
        #     try:
        #         sentiment_class = np.sign(int(res["sentences"][0]["sentimentValue"])-2)
        #     except:
        #         continue
        #     finally:
        #         html_text += sentiment_highlighter(sentence, sentiment_class, threshold)

        # To extract both class label and predicted probabilities, use the code below
        article = tag_output
        # Will need to set access for the server to read and write in this directory
        base_dir = '/home/hareesh/IPython/NLP/stanford-corenlp-full-2016-10-31_1/'
        
        with open(base_dir+"input_doc.txt", "w") as text_file:
            text_file.write(re.sub(r'[^\x00-\x7F]+',' ', article).strip())

        os.system("cd " + base_dir + "&& java -cp '*' -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -output PROBABILITIES -file " + base_dir + "input_doc.txt >  " + base_dir + "output_doc.txt")
        with open(base_dir+"output_doc.txt", 'r') as text_file:
            parse_output = text_file.read()
            scores = [line for line in parse_output.split('\n') if " 0:" in line]

            lines = parse_output.split('\n')
            sentence_list = []
            for k in range(len(lines)):
                if lines[k].startswith("  0:"):
                    sentence_list.append(lines[k-2])

            threshold_satisfy = False
            for j in range(len(sentence_list)):
                class_label = np.argmax(map(float, scores[j].split("  ")[-5:]))
                if class_label == 4 or (class_label == 3 and float(scores[j].split("  ")[5]) > float(threshold)):
                    threshold_satisfy = True
                elif class_label == 0 or (class_label == 1 and float(scores[j].split("  ")[3]) > float(threshold)):
                    threshold_satisfy = True
                html_text += sentiment_highlighter(sentence_list[j], class_label, threshold_satisfy)
 

    return html_text


# In[23]:

def highlighter(input_text, classifer, threshold):
    # NER Tagging
    tag_output = tagger.tag_text(input_text)
    # Remove /O tags
    tag_output = re.sub("/O ", " ", tag_output) # Need to fix the issues for punctuations. Example: "/O ./O" to be replaced with '. ' Similarly for , ; etc [Group match and replace]

    tag_output = bracket_corrector(tag_output)
    
    # Next, do sentence level sentiment annotation
    sentiment_output = sentiment_classifier(tag_output, classifer, threshold)
    
    clean_html = entity_highlighter(sentiment_output)
    return clean_html


# In[ ]:
# input_text = """
# A tractor trailer full of beer drove itself down Colorado's I-25 last week with nobody behind the wheel. Uber Technologies Inc. and Anheuser-Busch InBev NV teamed up on the delivery, which they said is the first time a self-driving truck had been used to make a commercial shipment.
# "We wanted to show that the basic building blocks of the technology are here; we have the capability of doing that on a highway," said Lior Ron, the president and co-founder of Uber's Otto unit. "We are still in the development stages, iterating on the hardware and software."
# The death of a driver using Tesla Motors Inc.'s autopilot system in May has focused political attention on self-driving vehicles and hastened calls for regulations to keep pace with the technological advances. The U.S. Transportation Department released policy guidelines for autonomous driving, which acknowledged the technology's life-saving potential while warning of a world of "human guinea pigs."
# """
# print highlighter(input_text, "TextBlob_Lexicon")