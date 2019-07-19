import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
from wordcloud import STOPWORDS

#Load the dataset
dataset = pd.read_csv('/home/jamcey/Downloads/Exploring Text Data/tweets.csv', encoding = 'ISO-8859-1')


def gen_freq(text):

    word_list = []
    for tw_words in text.split():
        word_list.extend(tw_words)

    word_freq = pd.Series(word_list).value_counts()


    word_freq[:20]

    return word_freq


#gen_freq(dataset.text.str)
word_freq = gen_freq(dataset.text.str)

wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

def clean_text(text):

    text = re.sub(r'RT', '', text)

    text = re.sub(r'&amp;', '&', text)

    text = re.sub(r'[?!.;:,#@-]', '', text)

    clean = re.compile('<.*?>')
    text = re.sub(clean,'', text)

    text = text.lower()
    return text

#to check for stopwords
print(STOPWORDS)


text = dataset.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 14))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
print(text)



