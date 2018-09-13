import tweepy
from textblob import TextBlob

consumer_key = 'AMbpB35Z65C28a5iFRU5a0Sc0'
consumer_secret = 'O7SZNwjx7N79Ipxuh74XgQew1LfUegjNvQ64zN06yIli5E2Teo'

access_token = '1040207435871412226-tNEalaGJLlGYhPMdLRsyvPTnJcXkuJ'
access_token_secret = 'qngFGYtpg2qPBKbdWW0dmeAFTDwyIdRxWkPq94qV0qllo'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

# polarity measures how positive or negative some text is
# subjectiviy meauser how much of an opinion it is versus how factual
for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

# consumer key
# set access key by using consumer key
# access to API
# search certain words in api
# for each tweet print the text
# make each text to analyze as variable
# analyze sentiment of variable holding text