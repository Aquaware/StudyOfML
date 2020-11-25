from janome.tokenizer import Tokenizer
import zipfile
import os.path
import urllib.request as request
import re
from gensim.models import Word2Vec



def ex1():
    tokenizer = Tokenizer()
    ma = tokenizer.tokenize('すももももものうち')
    for n in ma:
        print(n)

def download(url):
    zip_file = 'story.zip'
    text_file = 'fufuga_sakka.txt'
    request.urlretrieve(url, zip_file)
    try:
        zip = zipfile.ZipFile(zip_file)
    except Exception as e:
        print(e)

    #zip.extractall('./data/')
    f = zip.open(text_file)
    t = f.read()
    text = t.decode('shift_jis')
    f.close()
    return text

def body(text):
    t1 = re.split(r'\-{5,}', text)[2]
    t2 = re.split(r'底本：', t1)[0]
    t3 = t2.replace('|', '')
    t4 = re.sub(r'《.+?》', '', t3)
    t5 = re.sub(r'[#.+?]', '', t4)
    return t5

def parse(text):
    lines = text.split('\r\n')
    tokenizer = Tokenizer()
    results = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = tokenizer.tokenize(line)
        r = []
        for token in tokens:
            if token.base_form == '*':
                w = token.surface
            else:
                w = token.base_form
            hinshi = token.part_of_speech.split(',')[0]
            if hinshi in ['名詞', '形容詞', '動詞', '記号']:
                r.append(w)
        rl = (' '.join(r)).strip()
        results.append(rl)
        return results

def ex2():
    t1 = download('https://www.aozora.gr.jp/cards/000311/files/3153_ruby_10790.zip')
    text = body(t1)
    words = parse(text)
    print(words)
    model = Word2Vec(words, sg=1, size=100, window=5, min_count=1)
    for item in model.most_similar('夫婦'):
        print(item)





    print(results)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ex2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
