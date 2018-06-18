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
finished_files_dir = "finished_files"

num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506
BOS = '<s>'
EOS = '</s>'

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

def fix_missing_period(line):
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."

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

    abstract = ' '.join("%s %s %s" % (BOS, sent, EOS) for sent in highlights)

    return article, abstract

def write2file(url_file, out_file):
    ulr_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    outname = os.path.join(out_file, '.txt')
    with open(outname, 'at') as writer:
        for idx, s in enumerate(story_fnames):

            #check file
            if os.path.isfile(os.path.join(cnn_stories_dir,s)):
                story_file = os.path.join(cnn_stories_dir, s)
            elif os.path.isfile(os.path.join(dm_stories_dir,s)):
                story_file = os.path.join(dm_stories_dir,s)
            else:
                check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
                check_num_stories(dm_stories_dir, num_expected_dm_stories)
                raise Exception("Error")

            #getting art-abs pair
            article, abstract = get_art_abs(story_file)

            #store in one file
            writer.write(' '.join(article) + '\t' + ' '.join(abstract) + '\n')



if __name__ == '__main__':
    cnn_stories_dir = sys.argv[1]
    dm_stories_dir = sys.argv[2]

    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    write2file(all_train_urls, os.path.join(finished_files_dir, "train.txt"))
    wrtie2file(all_val, os.path.join(finished_files_dir, "val.txt"))
    wrtie2file(all_test, os.path.join(finished_files_dir, "test.txt"))
