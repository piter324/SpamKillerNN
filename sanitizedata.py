import csv
import re
from nltk.corpus import stopwords
import nltk.stem

def get_csv_to_array(filename):
    csv_arr = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_arr.append(row)
    # print(csv_arr[0])
    return csv_arr

def sanitize(text):
    text = text.lower() # text to lowercase
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\s\s+',' ',text) # replace multiple spaces with just one
    text_arr = []
    for word in text.split(' '):
        if(len(word) <= 20): text_arr.append(word)
    text_arr = remove_stopwords(text_arr)
    text_arr = do_stemming(text_arr)
    return text_arr

def remove_stopwords(text_arr):
    stopwords_arr = stopwords.words("english")
    new_text_arr = text_arr[:] # copy first text_arr
    for word in text_arr:
        if word in stopwords_arr:
            new_text_arr.remove(word)
    return new_text_arr

def do_stemming(text_arr):
    stemmer = nltk.stem.SnowballStemmer("english") # english stemmer works better than porter stemmer
    for i in range(len(text_arr)):
        text_arr[i] = stemmer.stem(text_arr[i])
    return text_arr


def prepare_csv(filename):
    csv_arr = get_csv_to_array(filename)
    for i in range(len(csv_arr)):
        # print("------")
        csv_arr[i][2] = sanitize(csv_arr[i][2])
        csv_arr[i][4] = sanitize(csv_arr[i][4])
        # print(csv_arr[i])
    return csv_arr

if __name__ == '__main__':
    csv_input = prepare_csv("csv01.csv")
    print(csv_input)
