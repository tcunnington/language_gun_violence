## Language of gun violence

### Introduction

The goal of this project is to investigate the language used in the news in the wake of gun violence. Unfortunately gun violence is becoming a common occurrence in America and therefore there are many documents on such events. I want to use this to fond out more about how we talk about gun violence.

I will primarily look at term frequency analysis for various sets of news stories that are reporting on or discussing (in the case of an opinion piece) these attacks. In addition to obvious terms such as “shooting” I want to look at term frequency for more culturally loaded terms such as “hopes and prayers”, “gun control”, and “mental illness”. I plan to see how those frequencies change over time in the weeks following an incident, as well as differences between incidences. This might answer, for instance, is there a specific lifecycle for the national conversation following a tragedy? Does the language of the conversation change depending on the race or religion of the shooter? 

### Data set:

I am using two data sets for this project. The primary one is approximately 150,000 news stories from 2016 and 2017 that I obtained from Kaggle.com (link below). It provides the story content and title, as well as some other information. The full data set is over 600mb.

The second dataset is a list of mass shootings in US from Mother Jones. It gives basic where/when/who types of information, plus background information on the shooter and weapon they used. It also cites new stories as sources. I use these data to help me find particular new events in the larger dataset, as well as to provide some extra context for a given event.

Links: https://www.kaggle.com/snapcrack/all-the-news, https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/


### Methods:

##### Preprocessing

In order to calculate term frequencies I first had to preprocess the text using standard NLP techniques. I removed common words and used lemmatization to reduce words like (‘mouse’, ‘mice’) to ‘mouse’ (I also tried stemming but haven’t conclusively decided which is better). This reduces the space of words in a doc and so that you count different variants of a word as the same word. With this step complete you can now tokenize documents to get “terms”. I mostly tokenized by single words to get unigrams, but also used bi-grams.

I then used scikit-learn to calculate two types of count vectors: term frequency (TF) and inverse document frequency (TFIDF). In both cases a document is represented by a vector, where each dimension (index) corresponds to a term. When calculating TF the values are the raw counts of a given term; and when calculating TFIDF the values are TFIDF measure: term frequency divided by the document frequency. Here also there was additional filtering of terms when I considered the entire corpus: I rejected terms with document frequency > 95%  and < 2 (absolute).

##### Finding stories for an event

Since most of the stories in the dataset were not related to shootings I had to find stories that were similar to ones I could find. To do this I calculated the TFIDF vectors for all stories in the date range 7-14 days after the shooting and then calculated the cosine similarity of the stories to each other. I used this metric stories similar to a given story I knew was on the subject I cared about. This ended up working very well and allowed me to find sets of stories on a given topic automatically. 

### Takeaways:

insert venn diagram of terms
discuss

insert timeline plot
discuss

anything else?

### Future Plans:

- More data! Fill in years with poor coverage. Perhaps another sort of corpus altogether: politicians’ speeches/statements, NRA statements, twitter, etc.
- Compare language of different news sources
- More time series plots. Including other big stories and word usage over months and years
- Lots of experimentation with visualizations, and at least one interactive plot
- Use Mother Jones data set to add more context or aid in further group/slice/filter experimentation