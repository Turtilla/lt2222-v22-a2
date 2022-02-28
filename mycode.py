import gzip
import random


def sample_lines(zipped_file, lines=100000):
    unzipped_file = gzip.open(zipped_file, "rb")
    contents = unzipped_file.readlines()
    random_slice = random.randint(0, (len(contents)-lines))
    print(random_slice)
    
    return contents[random_slice:random_slice+lines]


if __name__ == "__main__":
    sampled_lines = sample_lines(r"C:\Users\turti\OneDrive\Dokumenty\GitHub\machine_learning\UN-english.txt.gz", lines=100000)
    print(len(sampled_lines))
    print(sampled_lines[40000:40010])