import sys
import os
import collections
import hashlib
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens

#This script aims at preprocessing raw data into the format that can be the input for transform.py
dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.','!','?','...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_val.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"
chunk_dir = os.path.join(finished_files_dir, "chunked")

num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

CHUNK_SIZE = 1000

SOS = '<s>'
EOS = '</s>'

def write2file(url_file, out_file, makevocab=False):
    ulr_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = count_tokens("")

    for idx, s in enumerate(story_fnames):

        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir,s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir,s)):
            story_file = os.path.join(dm_tokenized_stories_dir,s)
        else:
            check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
            check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
            raise Exception("Error")


        article, abstract = get_art_abs(story_file)

        if makevocab:
            vocab_counter(article, counter = vocab_counter)
            vocab_counter(abstract, counter = vocab_counter)

        outname = os.path.join(out_file, '%s.txt' % (set_name))
        with open(outname, 'at') as writer:
            writer.write(str(idx) + '\t' + ' '.join(article) + '\t' + ' '.join(abstract) + '\n')
            # writer.write(article + '\n')
            # writer.write("abstract:\n")
            # writer.write(abstract + '\n')

        cnt += 1

    if makevocab:
        return vocab_counter



def read_text_file(text_file):
    lines = []
    with open(text_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

#find correponding train,val,test
def hashhex(s):
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def get_url_hashes(url_list):
    return [hashhex(url) for url in url_lsit]


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception("Error")



def tokenize_stories(stories_dir, tokenize_stories_dir):
    stories = os.listdir(infile)
    cnt = 0
    for s in stories:
        if s == '.DS_Store':
            continue
        in_path = os.path.join(infile,s)
        out_path = os.path.join(outfile, s)
        tokens = CorpusDataset(in_path, tokenizer = SpacyTokenizer())

        with open(out_path, 'w') as w:
            for tk in tokens:
                w.write(' '.join(tk) + '\n')

    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenize_stories_dir))

    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    lines = [lines.lower() for line in lines]

    lines = [fix_missing_period(line) for line in lines]

    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True;
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)

    abstract = ' '.join("%s %s %s" % (SOS, sent, EOS) for sent in highlights)

    return article, abstract

def fix_missing_period(line):
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."

if __name__ == '__main__':
    cnn_stories_dir = sys.argv[1]
    dm_stories_dir = sys.argv[2]

    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)

    if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
    if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
    tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

    write2file(all_train_urls, os.path.join(finished_files_dir, "train"))
    wrtie2file(all_val, os.path.join(finished_files_dir, "val"))
    wrtie2file(all_val, os.path.join(finished_files_dir, "test"))
