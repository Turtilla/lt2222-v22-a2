import gzip
import random
import math
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def sample_lines(zipped_file, lines=100000):
    unzipped_file = gzip.open(zipped_file, "rb")
    contents = unzipped_file.readlines()
    random_slice = random.randint(0, (len(contents)-lines))
    lines = contents[random_slice:random_slice+lines]
    sampled_lines = []
    for line in lines:
        decoded_line = line.decode('utf8').strip()
        sampled_lines.append(decoded_line)
    
    return sampled_lines

def process_sentences(sampled_lines):
    tokenizer = WordPunctTokenizer()
    preprocessed_sentences = []
    for line in sampled_lines:
        tokenized = tokenizer.tokenize(line)
        preprocessed_sentences.append(tokenized)

    tagged_sentences = []
    for sentence in preprocessed_sentences:
        tagged_sentence = nltk.pos_tag(sentence)
        tagged_sentences.append(tagged_sentence)

    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in tagged_sentences:
        sentence_2 = []
        for word, POS in sentence:
            word = word.lower()
            if word in stop_words:
                continue
            elif word.isalpha() == False:
                continue
            elif word == ".":
                continue
            elif len(word) < 2:
                continue
            else:
                pair = (word, POS)
                sentence_2.append(pair)
        processed_sentences.append(sentence_2)

    return processed_sentences

def create_samples(processed_sentences, samples=50000):
    five_grams = []
    for sentence in processed_sentences:
        five_gram = list(ngrams(sentence, 5))
        for gram in five_gram:
            five_grams.append(gram)

    random_slice = random.randint(0, (len(five_grams)-samples))
    samples = five_grams[random_slice:random_slice+samples]

    all_samples = []
    for pair1, pair2, pair3, pair4, pair5 in samples:
        word1 = get_ending(pair1[0]) + '_1'
        word2 = get_ending(pair2[0]) + '_2'
        word4 = get_ending(pair4[0]) + '_4'
        word5 = get_ending(pair5[0]) + '_5'
        POS = pair3[1]

        if "V" in POS:
            verb = 1
        else:
            verb = 0
    
        sample = ((word1, word2, word4, word5), verb)
        all_samples.append(sample)        

    return all_samples

def get_ending(word):
    ending = word[-2:]
    return ending

def create_df(all_samples):
    possible_columns = []
    possible_columns_set = set()
    for sample in all_samples:
        endings = sample[0]
        for ending in endings:
            if ending not in possible_columns_set:
                possible_columns.append(ending)
                possible_columns_set.add(ending)
    possible_columns.append('verb')
    
    random.shuffle(all_samples) # to make sure that there is less bias
    sample_vectors = []
    for sample in all_samples:
        endings = sample[0]
        ifverb = sample[1]
        vector = []
        for label in possible_columns:
            if label != 'verb':
                if label in endings:
                    vector.append(1)
                else:  # if label not in endings
                    vector.append(0)
        vector.append(ifverb)
        sample_vectors.append(vector)

    vector_array = np.array(sample_vectors)
    vector_df = pd.DataFrame(vector_array, columns=possible_columns)

    return vector_df

def split_samples(fulldf, test_percent=20):
    column_number = len(fulldf.columns)
    print(column_number)
    full_X = fulldf.iloc[:, 0:column_number-2]
    full_y = fulldf.iloc[:, column_number-1]
    
    row_number = len(fulldf)
    cutoff = math.ceil(row_number * (test_percent / 100))
    
    train_X = full_X[cutoff:]
    train_y = full_y[cutoff:]
    test_X = full_X[:cutoff]
    test_y = full_y[:cutoff]

    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    sampled_lines = sample_lines(r"C:\Users\turti\OneDrive\Dokumenty\GitHub\machine_learning\UN-english.txt.gz", lines=10000)
    print(len(sampled_lines))
    print(sampled_lines[4000:4010])

    processed_sentences = process_sentences(sampled_lines)
    print(processed_sentences[4000:4010])

    all_samples = create_samples(processed_sentences, samples=5000)
    print(all_samples[2500:2510])

    fulldf = create_df(all_samples)
    print(fulldf[2500:2510])

    train_X, train_y, test_X, test_y = split_samples(fulldf, test_percent=20)
    print(len(train_X), len(train_y), len(test_X), len(test_y)) 