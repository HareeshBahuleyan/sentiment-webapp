# Sentiment and Named Entity Highlighter
A web application in Python Django that highlights:

1. Sentiments (Positive/Negative/Neutral) at the sentence level  
2. Named Entities (Organization/Person/Location)

The backend for the named entity is <a href="http://stanfordnlp.github.io/CoreNLP/ner.html">Stanford NER</a>.
For sentiment analysis, 4 models have been used:

1. <a href="http://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers">TextBlob Lexicon (based on Pattern)</a>
2. <a href="http://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers">TextBlob Naive Bayes (based on NLTK)</a>
3. <a href="http://www.nltk.org/api/nltk.sentiment.html">NLTK Vader</a> 
4. <a href="http://nlp.stanford.edu/sentiment/"> Stanford Core NLP Deep Learning </a>

### Requirements
Python packages: textblob, nltk, ner 
Download and extract the CoreNLP Jar files for <a href="http://nlp.stanford.edu/software/stanford-english-corenlp-2016-10-31-models.jar"> Sentiment</a> and for <a href="http://nlp.stanford.edu/software/stanford-ner-2015-12-09.zip">NER</a>

### Running
Two servers need to be running in the background for CoreNLP to function. <br>
<div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">cd stanford-english-corenlp-2016-10-31-models/ 
java -mx5g -cp &quot;*&quot; edu.stanford.nlp.pipeline.StanfordCoreNLPServer 
cd stanford-ner-2015-12-09 
java -mx1000m -cp &quot;stanford-ner.jar:lib/*&quot; edu.stanford.nlp.ie.NERServer -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -port 9191 
</pre></div>

Once this is done, start the django app by executing the following commands: <br>
<div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;"><pre style="margin: 0; line-height: 125%">cd sentimentapp/
python manage.py runserver
</pre></div>
