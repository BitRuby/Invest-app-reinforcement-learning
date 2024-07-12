#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
def show_table(data, fields):
    df = pd.DataFrame(data=data)
    df = df[fields]
    df.sort_values(by=fields[0], inplace=True)
    return df


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
def print_plots(dataframe_a, dataframe_b, field_a, field_b, custom_a_name='', custom_b_name=''):
    dataframe_a['date'] = pd.to_datetime(dataframe_a['date'])
    dataframe_b['date'] = pd.to_datetime(dataframe_b['date'])
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    plt.plot(dataframe_a['date'], dataframe_a[field_a])
    plt.title(custom_a_name if custom_a_name else field_a)
    plt.xlabel("date")
    plt.ylabel(custom_a_name if custom_a_name else field_a)
    plt.subplot(1,2,2)
    plt.plot(dataframe_b['date'], dataframe_b[field_b], color="orange")
    plt.title(custom_b_name if custom_b_name else field_b)
    plt.xlabel("date")
    plt.ylabel(custom_b_name if custom_b_name else field_b)
    plt.tight_layout()
    plt.show()


# In[1]:


def print_plot(dataframe, field, custom_name=''):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    plt.plot(dataframe['date'], dataframe[field])
    plt.title(custom_name if custom_name else field)
    plt.xlabel("date")
    plt.ylabel(custom_name if custom_name else field)
    plt.tight_layout()
    plt.show()


# In[3]:


def sma(data, window_size):
    return data.rolling(window=window_size).mean()


# In[4]:


import re
def apply_transformations(transformations, text):
    if "REMOVE_USERNAMES" in transformations:
        remove_usernames_pattern = r'@\S+'
        text = re.sub(remove_usernames_pattern, "", text)
    if "REMOVE_URLS" in transformations:
        remove_urls_pattern = r'http\S+'
        text = re.sub(remove_urls_pattern, "", text)
    if "REMOVE_PUNCTUATION_MARKS" in transformations:
        remove_pun_pattern_1 = r"(?<!\d)\.(?!\d)|[^\w\s.']"
        remove_pun_pattern_2 = r"'"
        remove_pun_pattern_3 = r"\s\s+"
        sub1 = re.sub(remove_pun_pattern_1, " ", text)
        sub2 = re.sub(remove_pun_pattern_2, "", sub1)
        text = re.sub(remove_pun_pattern_3, " ", sub2)
    if "TEXT_TO_LOWER" in transformations:
        text = text.lower()
    if "REMOVE_SHORT_WORDS" in transformations:
        remove_short_pattern = r'\b\w{1,2}\b'
        text = re.sub(remove_short_pattern, '', text)
    return text


# In[5]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def sentiment_analyze(text):
    return analyzer.polarity_scores(text)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')

def text_vectorizer(text):
    tokenized = nltk.tokenize.word_tokenize(text)
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(tokenized).toarray()


# In[13]:


def transform_tweet_list(tweet_list, transform_types):
    transformed_tweet_list = [{
        "date": obj["date"],
        "content": 
            apply_transformations(transform_types, obj["content"]),
        **sentiment_analyze(apply_transformations(transform_types, obj["content"]))
    } for obj in tweet_list]
    return transformed_tweet_list


# In[8]:


from datetime import datetime

def merge_and_accumulate(data):
    daily_data = {}
    for entry in data:
        datetime_object = datetime.strptime(entry["date"][:-1], '%Y-%m-%dT%H:%M:%S.%f')
        date_obj = datetime_object.strftime('%Y-%m-%d')
        if date_obj in daily_data:
            daily_data[date_obj]["pos"].append(entry["pos"])
            daily_data[date_obj]["neg"].append(entry["neg"])
            daily_data[date_obj]["neu"].append(entry["neu"])
            daily_data[date_obj]["compound"].append(entry["compound"])
        else:
            daily_data[date_obj] = {
                "pos": [entry["pos"]],
                "neg": [entry["neg"]],
                "neu": [entry["neu"]],
                "compound": [entry["compound"]]
            }
    mapped_array = []
    for date, info in daily_data.items():
        mapped_array.append({
            "date": date,
            "pos_avg": sum(info["pos"]) / len(info["pos"]),
            "neg_avg": sum(info["neg"]) / len(info["neg"]),
            "neu_avg": sum(info["neu"]) / len(info["neu"]),
            "compound_avg": sum(info["compound"]) / len(info["compound"])
        })
    return mapped_array


# In[9]:


import numpy as np

def apply_min_max(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
def apply_standarization(data):
    return (data - np.mean(data)) / np.std(data)


# In[10]:


def choose_other(field_str, trend):
    if trend == "higher":
        return f"{field_str}_lower"
    elif trend == "lower":
        return f"{field_str}_higher"


# In[11]:


import numpy as np
def create_trends(field_str):
    prev = None
    trend = None
    def map_obj(obj):
        nonlocal prev
        nonlocal trend
        if np.isnan(obj[field_str]):
            if trend == None:
                obj[f"{field_str}_lower"] = 0
                obj[f"{field_str}_higher"] = 0
            else:
                obj[f"{field_str}_{trend}"] = 1
                obj[choose_other(field_str, trend)] = 0
        else:
            if prev == None:
                obj[f"{field_str}_lower"] = 0
                obj[f"{field_str}_higher"] = 0
            else:
                if obj[field_str] < prev:
                    obj[f"{field_str}_lower"] = 1
                    obj[f"{field_str}_higher"] = 0
                    trend = "lower"
                else:
                    obj[f"{field_str}_lower"] = 0
                    obj[f"{field_str}_higher"] = 1
                    trend = "higher"
            prev = obj[field_str]
        return obj
    return map_obj


# In[12]:


import pandas as pd
def merge_two_df_by_dates(df1, df2, how='right'):
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    df1['date'] = df1['date'].dt.tz_localize(None)
    df2['date'] = df2['date'].dt.tz_localize(None)
    merged_df = pd.merge(df1, df2, on='date', how=how)
    return merged_df

