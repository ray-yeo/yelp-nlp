import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

import nltk
# nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

# https://towardsdatascience.com/text-mining-and-sentiment-analysis-for-yelp-reviews-of-a-burger-chain-6d3bcfcab17b


"""
Preprocessing Data
"""

restaurants_df = pd.read_csv('clean_restaurants1.csv')

italian_restaurants_df = restaurants_df[restaurants_df['Italian'] == True]
italian_restaurant_id_list = italian_restaurants_df['business_id']  

mexican_restaurants_df = restaurants_df[restaurants_df['Mexican'] == True]
mexican_restaurant_id_list = mexican_restaurants_df['business_id']  

print('created restaurant dfs')

with open('phillyReviews.json') as f:
    print("read in file")
    for jsonObj in f:
        reviews = json.loads(jsonObj)

# dataframe of all reviews
df = pd.DataFrame(reviews)

print(len(df))

# get just the restaurants
italian_df = df[df['business_id'].isin(italian_restaurant_id_list)]
mexican_df = df[df['business_id'].isin(mexican_restaurant_id_list)]
print(len(italian_df))
print(len(mexican_df))


# working with a small subset for runtime
# df = df.sample(frac = 0.1)
# print(len(df))
# print(len(reviews))


# 5 star reviews
df_five_star_italian = italian_df[italian_df['stars'] == 5]
df_five_star_mexican = mexican_df[mexican_df['stars'] == 5]
print(df_five_star_italian)
print(df_five_star_mexican)


# 1 star reviews
df_one_star_italian = italian_df[italian_df['stars'] == 1]
df_one_star_mexican = mexican_df[mexican_df['stars'] == 1]


my_stop_words = set(stopwords.words('english'))



"""
Helper function to turn dataframes into dictionaries, in order to make visualizations
"""
def turn_dict(df):
    df_dict = dict(zip(df.bigram, df.frequency))
    return df_dict


"""
EXPLORATION
"""

def num_reviews_per_month(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    plt.plot(df['text'].resample('M').count())
    plt.xlabel('Year')
    plt.ylabel('Number of reviews')
    plt.title('Number of reviews per month')
    plt.show()

def distribution_customer_rating(df):
    ax = sns.barplot(data=df, x='stars', y='stars', estimator=lambda x: len(x) / len(df) * 100)
    ax.set(ylabel="Percent")
    plt.title('Distribution of Customer Rating')
    plt.show()

def avg_monthly_customer_rating(df):
    plt.plot(df['stars'].resample('M').mean())
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.title('Average Monthly Customer Rating')
    plt.ylim(0,5)
    plt.show()




"""
Find most common words
"""

def find_common_words(df):
    all_reviews_list = []
    # all_reviews_text = ""
    # my_stop_words.update(['go', 'definitely'])
    # print(my_stop_words)

    i = 0

    for review in df['text'].tolist():
        split_words = review.split()
        lower_words = [x.lower() for x in split_words]
        filtered_words = []
        for word in lower_words:
            if word not in my_stop_words:
                filtered_words.append(word)
        all_reviews_list += filtered_words
        i+=1
        print(i)


    # print(Counter(all_reviews_list).most_common(50))

    most_common = Counter(all_reviews_list).most_common(50)

    print(type(most_common))

"""
Find most common n-grams
"""
def common_n_grams(df):
    vect = CountVectorizer(stop_words=my_stop_words, ngram_range=(2,3))
    print("check1")
    bigrams = vect.fit_transform(df['text'])
    print("check2")
    bigram_df = pd.DataFrame(bigrams.toarray(), columns=vect.get_feature_names_out())
    print("check3")
    bigram_frequency = pd.DataFrame(bigram_df.sum(axis=0)).reset_index()
    print("check4")
    bigram_frequency.columns = ['bigram', 'frequency']
    print("check5")
    bigram_frequency = bigram_frequency.sort_values(by='frequency', ascending=False).head(50)
    print(bigram_frequency)
    print(type(bigram_frequency))
    df_dict = turn_dict(bigram_frequency)

    # create a word cloud out of it
    wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False, stopwords=my_stop_words).generate_from_frequencies(df_dict)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



"""
WORD 2 VEC
"""
def word_2_vec(df):
    selected_reviews = ' '.join(df.text)
    # split the long string into sentences
    tokenized_reviews = sent_tokenize(selected_reviews)
    # get tokens for each sentence
    token_cleaned = list()
    for sentence in tokenized_reviews:
        # When I parse natural sentences to extract entire words, I usually expand the allowed characters to also allow hyphens and apostrophes
        eng_word = re.findall(r'[A-Za-z\-]+', sentence)
        token_cleaned.append([i.lower() for i in eng_word if i.lower() not in my_stop_words])

    #sample
    print(tokenized_reviews[:3])
    print(token_cleaned[:3])

    #create model
    model_ted = Word2Vec(sentences=token_cleaned)
    print('food', model_ted.predict_output_word(['food'], topn=10))
    print('drinks', model_ted.predict_output_word(['drinks'], topn=10))
    print('service', model_ted.predict_output_word(['service'], topn=10))
    print('recommend', model_ted.predict_output_word(['recommend'], topn=10))
    print('price', model_ted.predict_output_word(['price'], topn=10))
    print('breakfast', model_ted.predict_output_word(['lunch'], topn=10))
    print('lunch', model_ted.predict_output_word(['lunch'], topn=10))
    print('dinner', model_ted.predict_output_word(['dinner'], topn=10))
    print('best', model_ted.predict_output_word(['best'], topn=10))
    # print('worst', model_ted.predict_output_word(['worst'], topn=10))
    print('too', model_ted.predict_output_word(['too'], topn=10))
    print('very', model_ted.predict_output_word(['very'], topn=10))
    print('love', model_ted.predict_output_word(['love'], topn=10))
    print('ambience', model_ted.predict_output_word(['ambience'], topn=10))
    print('expensive', model_ted.predict_output_word(['expensive'], topn=10))