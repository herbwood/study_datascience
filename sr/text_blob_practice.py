from textblob import TextBlob

# pip install textblob
# python -m textblob.download_corpora

sentence = "Presentation by donghyuk really sucks."
wiki = TextBlob(sentence)

print(wiki.tags)
print(wiki.words)

# -1 < sentiment < 1
print(wiki.sentiment.polarity)