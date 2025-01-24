import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
corpus = """I'm Faraz Maqsood. I'm doing bachelores in software engineering and now I'm learning generative AI."""
tokens = word_tokenize(corpus)
# print(tokens)
from nltk.tokenize import wordpunct_tokenize
