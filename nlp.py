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
Helper function to turn dataframes into dictionaries, in order to make visualizations
"""
def turn_dict(df):
    df_dict = dict(zip(df.ngram, df.frequency))
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

def distribution_customer_rating(df, category):
    ax = sns.barplot(data=df, x='stars', y='stars', estimator=lambda x: len(x) / len(df) * 100)
    ax.set(ylabel="Percent")
    plt.title('Distribution of Customer Rating: ' + category)
    plt.savefig('graphs/customer_rating.png')
    plt.show()

def avg_monthly_customer_rating(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    plt.plot(df['stars'].resample('M').mean())
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.title('Average Monthly Customer Rating')
    plt.ylim(0,5)
    plt.show()




"""
Find most common words
"""

def find_common_words(my_stop_words, df):
    all_reviews_list = []
    # all_reviews_text = ""
    # my_stop_words.update(['go', 'definitely'])
    # print(my_stop_words)

    i = 0

    for review in df['text'].tolist():
        split_words = review.replace('.', '').split()
        lower_words = [x.lower() for x in split_words]
        filtered_words = []
        for word in lower_words:
            if word not in my_stop_words:
                filtered_words.append(word)
        all_reviews_list += filtered_words
        i+=1
        # print(i)


    # print(Counter(all_reviews_list).most_common(50))

    most_common = Counter(all_reviews_list).most_common(50)
    # print(most_common)
    # print("making word cloud for common words")
    # # print(type(most_common))

    # wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False, stopwords=my_stop_words).generate_from_frequencies(dict(most_common))

    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")

    # plt.savefig('graphs/wordCloud.png')
    # plt.show()

    # print(type(most_common))
    return(most_common)

"""
Find most common n-grams
"""
def find_common_n_grams(my_stop_words, df):
    vect = CountVectorizer(stop_words=my_stop_words, ngram_range=(2,2))
    # print("check1")
    ngrams = vect.fit_transform(df['text'])
    # print("check2")
    ngram_df = pd.DataFrame(ngrams.toarray(), columns=vect.get_feature_names_out())
    # print("check3")
    ngram_frequency = pd.DataFrame(ngram_df.sum(axis=0)).reset_index()
    # print("check4")
    ngram_frequency.columns = ['ngram', 'frequency']
    print("check5")
    ngram_frequency = ngram_frequency.sort_values(by='frequency', ascending=False).head(50)
    print(ngram_frequency)
    # print(type(bigram_frequency))
    df_dict = turn_dict(ngram_frequency)



    # create a word cloud out of it
    wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False, stopwords=my_stop_words).generate_from_frequencies(df_dict)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.savefig('graphs/bigramCloud.png')
    plt.show()

    return df_dict



"""
WORD 2 VEC
"""
def word_2_vec(my_stop_words, df, words_to_check):
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
    # print(tokenized_reviews[:3])
    # print(token_cleaned[:3])

    #create model
    model_ted = Word2Vec(sentences=token_cleaned, window = 10)

    predictions = []
    for word in words_to_check:
        print(word)
        predictions.append((word[0], model_ted.predict_output_word(word, topn=10)))
    return predictions
    # print('food', model_ted.predict_output_word(['food'], topn=10))
    # print('drinks', model_ted.predict_output_word(['drinks'], topn=10))
    # print('service', model_ted.predict_output_word(['service'], topn=10))
    # print('recommend', model_ted.predict_output_word(['recommend'], topn=10))
    # print('price', model_ted.predict_output_word(['price'], topn=10))
    # print('breakfast', model_ted.predict_output_word(['lunch'], topn=10))
    # print('lunch', model_ted.predict_output_word(['lunch'], topn=10))
    # print('dinner', model_ted.predict_output_word(['dinner'], topn=10))
    # print('best', model_ted.predict_output_word(['best'], topn=10))
    # # print('worst', model_ted.predict_output_word(['worst'], topn=10))
    # print('too', model_ted.predict_output_word(['too'], topn=10))
    # print('very', model_ted.predict_output_word(['very'], topn=10))
    # print('love', model_ted.predict_output_word(['love'], topn=10))
    # print('ambience', model_ted.predict_output_word(['ambience'], topn=10))
    # print('expensive', model_ted.predict_output_word(['expensive'], topn=10))



def category_specific(restaurants_df, category):
    category_restaurants_df = restaurants_df[restaurants_df[category] == True]
    category_restaurant_id_list = category_restaurants_df['business_id']
    return category_restaurants_df, category_restaurant_id_list


def main():
    """
    Preprocessing Data
    """

    category_dict = {
        0: "Restaurants",
        1: "Food",
        2: "Nightlife",
        3: "Bars",
        4: "Sandwiches",
        5: "Pizza",
        6: "American (New)",
        7: "Breakfast & Brunch",
        8: "American (Traditional)",
        9: "Coffee & Tea",
        10: "Italian",
        11: "Chinese",
        12: "Fast Food",
        13: "Burgers",
        14: "Seafood",
        15: "Cafes",
        16: "Mexican",
        17: "Delis",
        18: "Event Planning & Services",
        19: "Salad",
        20: "Specialty Food",
        21: "Chicken Wings",
        22: "Bakeries",
        23: "Japanese",
        24: "Asian Fusion",
        25: "Vegetarian",
        26: "Caterers",
        27: "Desserts",
        28: "Sushi Bars",
        29: "Mediterranean",
        30: "Cheesesteaks",
        31: "Pubs"
    }

    restaurants_df = pd.read_csv('clean_restaurants1.csv')
    my_stop_words = set(stopwords.words('english'))
    my_stop_words.add('-')
    
    print('created restaurant dfs')

    with open('phillyReviews.json') as f:
        print("read in file")
        for jsonObj in f:
            reviews = json.loads(jsonObj)

    # dataframe of all reviews
    df = pd.DataFrame(reviews)

    # df = df.sample(frac = 0.01)

    # find_common_n_grams(my_stop_words, df)


    print(len(df))
    category = category_dict[30]
    category_restaurants_df, category_restaurant_id_list = category_specific(restaurants_df, category)
    category_df = df[df['business_id'].isin(category_restaurant_id_list)]

    df_five_star = category_df[category_df['stars'] == 5]
    df_one_star = category_df[category_df['stars'] == 1]

    # working with a small subset for runtime
    # df = df.sample(frac = 0.1)
    # print(len(df))
    # print(len(reviews))


    # print(my_stop_words)
    # num_reviews_per_month(category_df)
    # distribution_customer_rating(category_df, category)
    # avg_monthly_customer_rating(category_df)

    most_common_words = find_common_words(my_stop_words, category_df)
    # print(most_common_words)

    # sampled_df = category_df.sample(n=5000)
    # most_common_n_grams = find_common_n_grams(my_stop_words, sampled_df)
    # print(most_common_n_grams)



    five_star_results = word_2_vec(my_stop_words, df_five_star, most_common_words)

    print("Five Star Reviews")
    for word in five_star_results:
        print(word[0], ":", word[1])
    # print(five_star_results)
    one_star_results = word_2_vec(my_stop_words, df_one_star, most_common_words)
    print("One Star Reviews")
    for word in one_star_results:
        print(word[0], ":", word[1])
    # print(one_star_results)
    
if __name__ == "__main__":
    main()