## Language of gun violence

### Introduction

The goal of this project is to investigate the language used in the news in the wake of gun violence. Unfortunately gun violence is becoming a common occurrence in America and therefore there are many news stories on such events. I want to use these documents to find out more about how we talk about gun violence.

I will primarily look at term frequency analysis for various sets of news stories that are reporting on these attacks. In addition to obvious terms such as “shooting” I want to look at term frequency for more culturally loaded terms such as “hopes and prayers”, “gun control”, and “mental illness”. I plan to see how those frequencies change over time in the days following an incident, as well as differences between certain events. This might answer, for instance, is there a specific lifecycle for the national conversation following a tragedy? Does the language of the conversation change depending on the race or religion of the shooter? 

### Data set:

I am using two data sets for this project. The primary one is approximately 150,000 news stories from 2016 and 2017 that I obtained from Kaggle.com (link below). It provides the story content and title, as well as some other information. The full data set is over 600mb.

The second dataset is a list of mass shootings in US from Mother Jones. It gives basic where/when/who types of information, plus background information on the shooter and weapon they used. It also cites new stories as sources. I use these data to help me find particular new events in the larger dataset, as well as to provide some extra context for a given event.

Links: https://www.kaggle.com/snapcrack/all-the-news, https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/


### Methods:

##### Pre-processing

In order to calculate term frequencies I first had to preprocess the text using standard NLP techniques from the Natural Language Toolkit (nltk). I removed common words (stop words) and used lemmatization to reduce words like (‘mouse’, ‘mice’) to ‘mouse’ (I also tried stemming but haven’t conclusively decided which is better). This reduces the space of words in a doc and so that you count different variants of a word as the same word. With this step complete you can now tokenize documents to get “terms”. I mostly tokenized by single words to get unigrams, but also used bi-grams.

I then used scikit-learn to calculate two types of count vectors: term-frequency (TF) and term-frequency-inverse-document-frequency (TFIDF). In both cases a document is represented by a vector, where each dimension (index) corresponds to a term. When calculating TF the values are the raw counts of a given term; and when calculating TFIDF the values are the TFIDF measure: term frequency divided by the document frequency. Here also there was additional filtering of terms when I considered the entire corpus: I rejected terms with document frequency > 95%  and < 2 (absolute).

##### Finding stories for an event

Since most of the stories in the dataset were not related to shootings I had to find stories that were similar to ones I could find. To do this I calculated the TFIDF vectors for all stories in the date range 7-14 days after the shooting and then calculated the cosine similarity of the stories to each other. I used this metric to find stories similar to a story I knew was on the subject I cared about. This ended up working very well and allowed me to find sets of stories on a given topic automatically. 

### Analysis:


##### Top words

By comparing the top words between two events we can find out if there are language differences between the two events. The left circle contains the top words from the Orlando shooting and the right circle contains the top words from a shooting in Dallas, both in 2016.

![Comparing top words](img/tfidf_compare.png?raw=true)

Takeaways from this figure:
* Words you expect to be in a shooting story show overlap
* Words that make a story unique do not overlap
* Interestingly "attack" and "victim" are top words only in the Orlando story. This might suggest a different way of talking about events that are connected to terrorism.

##### Orlando timeline

I wanted to create some sense of word usage over time in the days after an attack. 
In order to do that I looked through the learned vocabulary for the Orlando shooting and started putting words into groups, such as putting the words 'gun', 'handgun', 'rifle', 'pistol', into a group called "gun vocab". 
I can then do analysis on the vocab groups, because that both reduces the considered vocabulary to terms (unigrams and bi-grams) that I care about, and it reduces the number of dimensions I need to consider.

Note: Gun vocab are synonyms for guns. 
Terror vocab are different words for terrorist, synonyms of ISIS, and words suggesting radical islam. 
Illness vocab are words commonly used to describe someone mentally unwell, such as "delusional".
The shooting vocab is language we certainly expect be in a story about a shooting, such as "shooter"".
 
![Orlando Timeline](img/orlando_timeline.png?raw=true)

Takeaways from this figure:

> On average terror vocab words were used about 3x as often as gun words

* Guns are not the focus of any of the articles for this event
* Mental illness is basically not discussed
* The shooter's ties to terrorism are mentioned frequently--perhaps there's typically not overlap between mental illness and terrorism discussions (Are we unwilling to consider terrorists to be mentally ill?)


### Future Plans:

- More data! Get stories to cover more recent events. Perhaps another sort of corpus altogether: politicians’ speeches/statements, NRA statements, twitter, etc.
- Compare language of different news sources, including sentiment.
- More time series plots. Including other big stories and word usage over months and years
- Lots of experimentation with visualizations, and at least one interactive plot
- Use Mother Jones data set to add more context or aid in further group/slice/filter experimentation