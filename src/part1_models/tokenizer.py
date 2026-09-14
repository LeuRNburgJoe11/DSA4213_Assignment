import re


#step 1: Tokenization function defined
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

def tokenize(text):
    '''Convert a string to a lowercase sequence of word/punctuation tokens.'''
    normalized = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(normalized)