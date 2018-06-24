import sys
import os
import collections
import hashlib
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import CorpusDataset
from gluonnlp.data import count_tokens

#This script aims at preprocessing raw data into the format that can be the input for transform.py
# dm_single_close_quote = u'\u2019'
# dm_double_close_quote = u'\u201d'
END_TOKENS = ['.','!','?','...', "'", "`", '"',  ")"]

all_train_urls = "/Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/data/train.txt"
all_val_urls = "/Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/data/val.txt"
all_test_urls = "/Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/data/test.txt"
finished_files_dir = "//Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/data/finished_files"
stories_dir = "/Users/Admin/Documents/DL/gluon-nlp-1/scripts/text_summarization/data/data"

num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506
BOS = '<s>'
EOS = '</s>'

def read_text_file(text_file):
    lines = []
    with open(text_file, 'r') as f:
        for line in f:
            lines.append(line.strip('\n').strip('\r'))
    # print lines
    return lines

#find correponding train,val,test
def hashhex(s):
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]

def fix_missing_period(line):
    if "@highlight" in line: return line
    if line == "" or line == " " or line == "  ": return ""
    if line[-1] in END_TOKENS: return line
    return line + " ."

def get_art_abs(story_file):
    lines = read_text_file(story_file)

    lines = [line.lower() for line in lines]

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


    # abstract = ' '.join("%s %s %s" % (BOS, sent, EOS) for sent in highlights)
    abstract = ' '.join(highlights)
    print abstract

    return article, abstract

def write2file(url_file, out_file):
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)

    # outname = os.path.join(out_file, '.txt')
    with open(out_file, 'at') as writer:
        for idx, s in enumerate(story_fnames):

            if os.path.isfile(os.path.join(stories_dir,s)):
                story_file = os.path.join(stories_dir, s)
            else:
                check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
                check_num_stories(dm_stories_dir, num_expected_dm_stories)
                raise Exception("Error")


            article, abstract = get_art_abs(story_file)

            writer.write(article + '\t' + abstract + '\n')



if __name__ == '__main__':
    # cnn_stories_dir = sys.argv[1]
    # dm_stories_dir = sys.argv[2]


    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)


    write2file(all_train_urls, os.path.join(finished_files_dir, "train.story"))
    write2file(all_val_urls, os.path.join(finished_files_dir, "val.story"))
    write2file(all_test_urls, os.path.join(finished_files_dir, "test.story"))
