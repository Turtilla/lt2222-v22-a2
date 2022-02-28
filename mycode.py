import gzip
import random
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.util import ngrams

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

    return samples




if __name__ == "__main__":
    sampled_lines = sample_lines(r"C:\Users\turti\OneDrive\Dokumenty\GitHub\machine_learning\UN-english.txt.gz", lines=100000)
    print(len(sampled_lines))
    print(sampled_lines[40000:40010])

    processed_sentences = process_sentences(sampled_lines)
    print(processed_sentences[40000:40010])

    all_samples = create_samples(processed_sentences, samples=50000)
    print(all_samples[25000:25010])