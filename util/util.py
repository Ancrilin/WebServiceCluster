from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords
from urlextract import URLExtract
import re
import string
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from datetime import timedelta
import time


class Preprocessor(object):
    def __init__(self, min_times):
        self.min_times = min_times
        self.wnl = WordNetLemmatizer()
        self.stem_tool = PorterStemmer()
        self.url_extractor = URLExtract()
        self.UK_check_list = ['u.k', 'u.k.', 'uk', 'UK', 'U.K', 'U.K.']
        self.USA_check_list = ['u.s.', 'u.s', 'U.S', 'U.S.', 'U.S.A', 'usa', 'USA', 'u.s.a', 'U.S.A.', 'u.s.a.']
        self.twoword_pattern = re.compile("[a-z][A-Z]{1}[a-z]", re.S)

        self.mark_words = []
        with open("tools/mark_words.txt", mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                self.mark_words.append(line.strip())

        #加入Fox停用词
        self.stopwords = []
        with open("tools/FoxStoplist.txt", mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                self.stopwords.append(line.strip())
        # 加入标点作为停用词
        self.stopwords.extend(list(string.punctuation))
        # 加入nltk标准停用词
        nltk_stopwords = list(stopwords.words('english'))
        self.stopwords.extend(nltk_stopwords)
        #统计词频的工具
        self.word_counter = Counter()


    def deal_text(self, text):
        #处理网址字符
        text = self.filter(text)
        if self.url_extractor.has_urls(text):
            sentence_urls = self.url_extractor.find_urls(text)
            for url in sentence_urls:
                text = text.replace(url, 'website-holder')

        doc_words = []
        for word in self.lemmatize_text(text):
            word = word.strip()

            match_string = re.findall(self.twoword_pattern, word)
            if len(match_string) != 0:
                for string in match_string:
                    words = word.replace(string, string[0] + ' ' + string[1:]).split(" ")
                    doc_words.extend(words)
            if word in self.stopwords:
                continue
            if word in self.UK_check_list:
                doc_words.append("uk")
            if word in self.USA_check_list:
                doc_words.append("usa")
            if word in self.mark_words:
                pattern = re.compile('[a-z]{1}-[a-z]{1}')
                match_string = re.findall(pattern, word)
                for string in match_string:
                    word = word.replace(string, string[0] + string[2:])
                    doc_words.append(word)
            #看是否为数字
            try:
                word_float = float(word)
                doc_words.append("#number#")
            except ValueError:
                pass
            if word == 'website-holder':
                doc_words.append("#website#")
            else:
                doc_words.append(word)
        #统一为小写
        for i, word in enumerate(doc_words):
            doc_words[i] = self.stem_tool.stem(word.lower())
        #统计词数
        self.word_counter.update(doc_words)

        return doc_words

    def lemmatize_text(self, text):
        wnl = WordNetLemmatizer()
        sentences = sent_tokenize(text, language='english')
        for sentence in sentences:
            for word, tag in pos_tag(word_tokenize(sentence, language='english')):
                if tag.startswith('NN'):
                    yield wnl.lemmatize(word, pos='n')
                elif tag.startswith('VB'):
                    yield wnl.lemmatize(word, pos='v')
                elif tag.startswith('JJ'):
                    yield wnl.lemmatize(word, pos='a')
                elif tag.startswith('R'):
                    yield wnl.lemmatize(word, pos='r')
                else:
                    yield wnl.lemmatize(word)

    def filter(self, text):
        """
        过滤掉无用的文本
        """
        if isinstance(text, float):
            return ''
        pattern = re.compile("\[This API is no longer available\.(.*?)\]", re.S)
        a = pattern.findall(text)
        if len(a) > 0:
            replace_item = "[This API is no longer available." + a[0] + ']'
            text = text.replace(replace_item, '').strip()
        else:
            text = text.strip()
        return text

    def read_data(self, csv_path):
        df = pd.read_csv(csv_path)
        all_process_cuts = []
        all_name = []
        all_tags = []
        all_labels = []
        label_map = dict()
        tag_map = {"UNK": 0}
        for i in tqdm(range(len(df))):
            row_data = df.iloc[i, :]
            APIName = row_data['name'].strip().lower()[:-4]  # 已经去掉了最后的API和空格
            # 处理tag
            tag_list = row_data['tags'].strip().lower().split(",")
            #生成tag_map
            for tag in tag_list:
                if tag not in tag_map:
                    tag_map[tag] = len(tag_map)
            # description
            raw_desc = row_data['description'].strip()
            desc_cuts = self.deal_text(raw_desc)
            # 处理category
            category = row_data['category'].strip()
            if category not in label_map:
                label_map[category] = len(label_map)

            all_name.append(APIName.strip())
            all_tags.append(tag_list)
            all_labels.append(category)
            all_process_cuts.append(desc_cuts)

        word_map = {'UNK': 0}
        for word, times in self.word_counter.items():
            if times < self.min_times:
                continue
            else:
                if word in word_map:
                    continue
                else:
                    word_map[word] = len(word_map)

        return all_name, all_process_cuts, all_labels, all_tags, word_map, tag_map, label_map


    def begin(self, csv_path, evaluate=False, old_word_map=None, old_tag_map=None, old_label_map=None):
        all_name, all_process_cuts, all_labels, \
        all_tags, get_word_map, get_tag_map, get_label_map = self.read_data(csv_path)

        all_encoded_tag = []
        all_encoded_category = []
        all_encoded_docs = []
        if not evaluate:
            word_map = get_tag_map
            tag_map = get_tag_map
            label_map = get_label_map
        else:
            assert old_word_map is not None and old_tag_map is not None and old_label_map is not None
            word_map = old_word_map
            tag_map = old_tag_map
            label_map = old_label_map

        #编码tag
        for doc_tag in all_tags:
            doc_encoded_tag = []
            for tag in doc_tag:
                if tag in tag_map:
                    doc_encoded_tag.append(tag_map[tag])
                else:
                    doc_encoded_tag.append(tag_map['UNK'])
            all_encoded_tag.append(doc_encoded_tag)

        #编码label
        for doc_label in all_labels:
            all_encoded_category.append(label_map[doc_label])

        #编码description
        for words in all_process_cuts:
            encoded_doc = []
            for word in words:
                if word in word_map:
                    encoded_doc.append(word_map[word])
                else:
                    encoded_doc.append(word_map['UNK'])
            all_encoded_docs.append(encoded_doc)

        #保存数据
        data_dict = dict()
        data_dict['all_name'] = all_name
        data_dict['all_process_cuts'] = all_process_cuts
        data_dict['all_labels'] = all_labels
        data_dict['all_tags'] = all_tags

        data_dict['all_encoded_category'] = all_encoded_category
        data_dict['all_encoded_tag'] = all_encoded_tag
        data_dict['all_encoded_docs'] = all_encoded_docs

        data_dict['word_map'] = word_map
        data_dict['label_map'] = label_map
        data_dict['tag_map'] = tag_map

        return data_dict


def build_dataset(filepath, min_times, evaluate=False, old_word_map=None, old_tag_map=None, old_label_map=None):

    preprocessor = Preprocessor(min_times)
    data_dict = preprocessor.begin(filepath, evaluate, old_word_map, old_tag_map, old_label_map)

    return data_dict



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



if __name__ == "__main__":
    # import pickle
    preprocessor = Preprocessor(1)
    # data_dict = preprocessor.begin('./webservice/data/20ClassesRawData.csv', './webservice/lda/20ClassesRawData_stem.input')
    all_name, all_process_cuts, all_labels, all_tags, word_map, tag_map, label_map = preprocessor.read_data('data/20ClassesRawData_API_cleanTag.csv')
    print(all_name, all_process_cuts, all_labels, all_tags, word_map, tag_map, label_map)

