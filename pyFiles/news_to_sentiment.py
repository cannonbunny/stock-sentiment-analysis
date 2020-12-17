# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from keras.utils import to_categorical


num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_gpu)
if num_gpu > 0: tf.debugging.set_log_device_placement(True)

# clean reuter
reuters = pd.read_csv('reuters_headlines.csv').dropna()
reuters = reuters[['Time', 'Headlines']]
reuters["Time"] = reuters["Time"].map(lambda time_str: datetime.strptime(time_str, "%b %d %Y"))
print("reuters:", min(reuters['Time']), max(reuters['Time']))

# clean cnbc
def clean_date_format(time_str):
  try:
    return datetime.strptime(time_str, "%d %B %Y")
  except:
    return datetime.strptime(time_str, "%d %b %Y")

cnbc = pd.read_csv('cnbc_headlines.csv').dropna()
cnbc = cnbc[['Time', 'Headlines']]
cnbc["Time"] = cnbc["Time"].map(lambda x: x.split(",")[1].strip())
cnbc["Time"] = cnbc["Time"].map(lambda x: x.replace("Sept ", "September ") if "Sept" in x else x)
cnbc["Time"] = cnbc["Time"].map(clean_date_format)
print("cnbc:", min(cnbc['Time']), max(cnbc['Time']))

# clean guardian
guardian = pd.read_csv('guardian_headlines.csv').dropna()
guardian = guardian[guardian.Time.str.len()!=6]
guardian = guardian[['Time', 'Headlines']]
guardian["Time"] = guardian["Time"].map(lambda time_str: datetime.strptime(time_str, "%d-%b-%y"))
print("guardian:", min(guardian['Time']), max(guardian['Time']))

# union and save
full_data = pd.concat([reuters, guardian, cnbc], ignore_index=True)
full_data.to_csv("news_full_data.csv", index=False)

print("Duration: ", min(full_data["Time"]), "->", max(full_data["Time"]))
print("Days between max_dt and min_dt: ", max(full_data["Time"]) - min(full_data["Time"]))
print("Days with datapoints: ", full_data["Time"].nunique())

def has_index(headline):
  keywords = ["U.S. economy", "S&P", "Dow", "Nasdaq", "Fed", "stocks", "Wall Street"]
  for word in keywords:
    if word in headline:
      return 1
  return 0

def get_daily_news(df):
  '''
  Input: df[Time, Headlines, has_index]
  Output: daily_news[date, text]
  '''
  dates = pd.date_range(min(full_data["Time"]),max(full_data["Time"])-timedelta(days=1),freq='d')
  news_list = []
  for date in dates:
    sub_df = df[df["Time"]==date]
    sub_df = sub_df[sub_df["has_index"]==1][:5] #前五篇
    news_combined = ". ".join(list(sub_df.Headlines))
    news_list.append(news_combined)
  daily_news = pd.DataFrame({"date": dates, "text": news_list})
  return daily_news

full_data["has_index"] = full_data["Headlines"].map(has_index)
daily_news = get_daily_news(full_data)


dateparse = lambda x: datetime.strptime(x, '%d-%m-%Y')
stock = pd.read_csv("GSPC.csv", parse_dates=['Date'], date_parser=dateparse)
stock = stock.iloc[:,[0,1]]
daily_news_2 = daily_news.join(stock.set_index('Date'), on='date')
daily_news_2 = daily_news_2.dropna(axis=0, subset=['y_Open'])
daily_news_2 = daily_news_2[['date','text']].reset_index(drop=True)


# Predict sentiment
model = keras.models.load_model('FISA_model3', compile=False)
pred = model.predict(daily_news_2["text"])
pred_2 = pd.DataFrame(data=pred, index=np.array(range(0, len(pred))), columns=np.array(range(0, 3)))
pred_2 = pred_2.join(daily_news_2)
pred_2 = pred_2[['date',0,1,2]]

pred_2.to_csv("FISA_pred.csv")
