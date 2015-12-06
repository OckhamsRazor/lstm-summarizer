import argparse
import codecs
import re
import xml.etree.ElementTree as ET
from os import mkdir, path
from shutil import rmtree


def padded_name(name):
    if name < 10:
        return "0000"+str(name)
    elif name < 100:
        return "000"+str(name)
    elif name < 1000:
        return "00"+str(name)
    elif name < 10000:
        return "0"+str(name)
    else:
        return str(name)


sent_boundary = ['!', '?', '.']
is_redirect = re.compile(r'^#REDIRECT', re.IGNORECASE)
wikipeida_href_or_label = re.compile(r'\{\{(?P<phrase>[^}]+)\}\}')
wikinews_href_or_media = re.compile(r'\[\[(?P<phrase>[^\]]+)\]\]')
external_link = re.compile(r'\[[^\]]+\]')
rule = re.compile(r'\{\|[^}\|]+\|\}')
def parse_content(content, abbrev):
    def wikipedia(match_obj):
        """
        For those in curly brackets
        """
        phrase = match_obj.group('phrase')
        tokens = phrase.split('|')
        if len(tokens) == 1 or tokens[0] != 'w': # phrase is a label
            return ""                            # discard it
        else:                 # phrase is a link to Wikipedia
            return tokens[-1] # keep its content

    def wikinews(match_obj):
        """
        For those in square brackets
        """
        phrase = match_obj.group('phrase')
        tokens = phrase.split('|')
        if len(tokens) == 1:   # phrase is a link to wikinews
            return phrase      # keep it
        elif len(tokens) == 2: # phrase is a link to wikipedia (or renamed)
            return tokens[-1]  # keep its content
        else:                  # otherwise, discard it
            return ""

    if is_redirect.match(content) is not None:
        return None

    content = content.split("{{haveyoursay}}")[0]
    content = content.split("==")[0]
    content = wikipeida_href_or_label.sub(wikipedia, content)
    content = wikinews_href_or_media.sub(wikinews, content)
    content = external_link.sub("", content)
    content = rule.sub("", content)

    content = content.replace("\n\n", "\n")
    content = content.lstrip()
    content = content.rstrip()
    for a in abbrev:
        # content = content.replace(" "+a, " "+a[:-1])
        no_coma = re.compile(r"\b"+re.escape(a), re.IGNORECASE)
        content = no_coma.sub(a[:-1], content)
    for s_b in sent_boundary:
        content = content.replace(s_b+" ", s_b+"\n")

    return content


news_brief = re.compile(r'^News Briefs:', re.IGNORECASE)
digest = re.compile(r'^Digest\/', re.IGNORECASE)
crosswords = re.compile(r'^crosswords', re.IGNORECASE)
def validate_title(title):
    if news_brief.match(title) is not None or digest.match(title) is not None or crosswords.match(title) is not None:
        return False
    return True


ns = {
    "xmlns": "http://www.mediawiki.org/xml/export-0.10/"
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wiki News Parser')
    parser.add_argument("-i", "--input", help="input file", required=True)
    parser.add_argument(
        "-O", "--output-dir", help="output folder", required=True
    )
    parser.add_argument(
        "-a", "--abbrev", help="Abbreviations", default="abbrev"
    )
    args = parser.parse_args()

    data = args.input
    output_dir = args.output_dir
    abbrev_file = args.abbrev

    if path.exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)

    abbrev = []
    with codecs.open(abbrev_file) as abbrev_fin:
        for line in abbrev_fin:
            word = line.rstrip()
            abbrev.append(word)


    root = ET.parse(data).getroot()
    count = 1
    for page in root.findall('xmlns:page', ns):
        if page.find('xmlns:ns', ns).text == '0':
            title = page.find('xmlns:title', ns).text
            content = page.find('xmlns:revision', ns).find('xmlns:text', ns).text

            content = parse_content(content, abbrev)
            if validate_title(title) and content is not None:
                with codecs.open(path.join(output_dir, padded_name(count)), 'w', 'utf-8') as fout:
                    fout.write(title)
                    fout.write('\n\n')
                    fout.write(content)
                    fout.write('\n')

                count += 1
