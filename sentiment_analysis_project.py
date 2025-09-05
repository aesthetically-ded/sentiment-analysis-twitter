import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

plt.style.use('fivethirtyeight')

# Step 1: Scrape Tweets using snscrape
query = "movies"
public_tweets = []
columns = ['Time', 'User', 'Tweet']
limit = 15000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(public_tweets) == limit:
        break
    else:
        if tweet.lang == 'en':
            public_tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(public_tweets, columns=columns)
df.to_csv("TweetsGenerated.csv", index=False)
print("Tweets collected:", len(df))

# Step 2: Clean / Preprocess Tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)       # removes @mentions
    text = re.sub(r'#', '', text)                   # removes # symbol
    text = re.sub(r'RT[\s]', '', text)             # removes retweets
    text = re.sub(r'https?:\/\/\S+', '', text)   # removes hyperlinks
    return text

df['Clean_Tweet'] = df['Tweet'].apply(cleanTxt)

# Step 3: Subjectivity & Polarity with TextBlob
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['Clean_Tweet'].apply(getSubjectivity)
df['Polarity'] = df['Clean_Tweet'].apply(getPolarity)

# Step 4: Categorize Sentiment
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
df.to_csv("TweetsProcessed.csv", index=False)

# Step 5: Visualizations
# WordCloud of all tweets
allWords = ' '.join([twts for twts in df['Clean_Tweet']])
wordCloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(allWords)
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Bar Chart of Sentiment Distribution
sent_counts = df['Analysis'].value_counts()
sent_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Step 6: Naive Bayes Classifier
X = df['Clean_Tweet']
y = df['Analysis']

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(text_counts, y, test_size=0.2, random_state=42)
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)
print("Naive Bayes Accuracy Score:", accuracy_score)
