import argparse
import codecs
from os import listdir, mkdir, path
from shutil import rmtree

import wordCount


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wiki news encoder')
    parser.add_argument(
        "-I", "--input-dir", help="raw texts (folder)", required=True
    )
    parser.add_argument(
        "-O", "--output-dir", help="encoded texts (folder)", required=True
    )
    parser.add_argument(
        "-d", "--dictionary", help="dictionary", required=True
    )
    parser.add_argument(
        "-t", "--title-oov-threshold", type=float, default=0.3
    )
    parser.add_argument(
        "-b", "--body-oov-threshold", type=float, default=0.15
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if path.exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)

    dictionary = args.dictionary
    title_oov_threshold = args.title_oov_threshold
    body_oov_threshold = args.body_oov_threshold
    if title_oov_threshold < 0 or title_oov_threshold > 1:
        print "Invalid title OOV threshold: %f" % title_oov_threshold
        exit(1)
    if body_oov_threshold < 0 or body_oov_threshold > 1:
        print "Invalid body OOV threshold: %f" % body_oov_threshold
        exit(1)

    OOV = 1
    d_arr = {}
    with open(dictionary, 'r') as d:
        word_id = 1
        for line in d:
            word = line.rstrip().lower()
            d_arr[word] = word_id
            word_id += 1

    for f in listdir(input_dir):
        doc = []
        with codecs.open(path.join(input_dir, f), 'r') as fin:
            # codecs.open(path.join(output_dir, f), 'w') as fout:

            for line in fin:
                sent = []
                tokens = line.rstrip().split()
                line = line.replace("`", "'")
                line = line.replace("''", '"')
                tokens = line.rstrip().split()

                for token in tokens:
                    words = wordCount.parse_token(token)
                    for word in words:
                        word = word.lower()
                        if word in d_arr:
                            w = d_arr[word]
                        else:
                            w = OOV
                        sent.append(w)

                doc.append(sent)

        if len(doc) > 2:
            title = doc[0]
            body = doc[2:]

            title_oov = title.count(OOV) / float(len(title))

            word_count = 0
            oov_count = 0
            for sent in body:
                word_count += len(sent)
                oov_count += sent.count(OOV)

            try:
                body_oov = oov_count / float(word_count)
            except ZeroDivisionError:
                continue

            # if title_oov < oov_threshold and body_oov < oov_threshold:
            if title_oov < title_oov_threshold and body_oov < body_oov_threshold:
                with codecs.open(path.join(output_dir, f), 'w') as fout:
                    for sent in doc:
                        for word in sent:
                            fout.write(str(word))
                            fout.write(' ')
                        fout.write('\n')

            # if w > 0:
            # if oov / float(w) > 0.2:
            # too_many_oovs.append(oov / float(w))
                            # total_oovs += 1
                        # token = d_arr[word]
                    # else:
                        # token = OOV

                        # fout.write(str(w))
                        # fout.write(' ')

                # fout.write('\n')

    # print total_words
    # print total_oovs
    # print len(listdir(input_dir))
    # print too_many_oovs
    # mean = sum(too_many_oovs) / len(too_many_oovs)
    # print mean
    # stdev = (sum((i-mean)**2 for i in too_many_oovs) / len(too_many_oovs))**0.5
    # print stdev
