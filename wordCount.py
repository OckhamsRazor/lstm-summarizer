import argparse
import codecs
from os import listdir, path


l_punct = ("(", '"')
r_punct = (',', '.', '!', '?', ';', '%', ':', ")")
def parse_token(token):
    """
    separates words and punctuations
    """
    words = []
    ends_with_quote = False

    if token.startswith(l_punct):
        words.append(token[0])
        token = token[1:]

    if token.endswith('"'):
        token = token[:-1]
        ends_with_quote = True

    if token.endswith(r_punct):
        words.append(token[:-1])
        words.append(token[-1])
    else:
        words.append(token)

    if ends_with_quote:
        words.append('"')

    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word Count')
    parser.add_argument(
        "-I", "--input-dir", help="raw texts (folder)", required=True
    )
    parser.add_argument(
        "-d", "--dictionary", help="output dictionary", required=True
    )
    parser.add_argument(
        "-min", "--minimal_count", type=int, default=5
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    d = args.dictionary
    minimal_count = args.minimal_count

    dictionary = {}
    for f in listdir(input_dir):
        with codecs.open(path.join(input_dir, f), 'r') as fin:
            blank_line_count = 0
            for line in fin:
                line = line.replace("`", "'")
                line = line.replace("''", '')
                tokens = line.rstrip().split()
                if len(tokens) == 0:
                    blank_line_count += 1
                    if blank_line_count > 1:
                        break
                    else:
                        continue

                for token in tokens:
                    words = parse_token(token)
                    for word in words:
                        word = word.lower()
                        if word not in dictionary:
                            dictionary[word] = 1
                        else:
                            dictionary[word] += 1

    # sorted_dict = reversed(sorted(dictionary.items(), key=itemgetter(1)))
    with codecs.open(d, 'w') as fout:
        fout.write("UNK\n")
        for word, count in dictionary.iteritems():
            if count >= minimal_count:
                fout.write(word+"\n")
