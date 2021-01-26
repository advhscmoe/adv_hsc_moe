import numpy
import json
import pickle as pkl
import random
import gzip
import shuffle

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            # return unicode_to_utf8(pkl.load(f))
            return pkl.load(f)


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

# dataset header: 
# top_cat_size	asin_hist	price_hist	overall_hist	
# brand_hist	unixReviewTime_hist	category_hist	overall	
# reviewerID	asin	reviewText	summary	unixReviewTime	
# category	description	title	pricetop_cat	next_cat	b
# rand	asin_overall_cnt_1	asin_overall_cnt_2	
# asin_overall_cnt_3	asin_overall_cnt_4	asin_overall_cnt_5	
# asin_overall_cnt	reviewer_overall_cnt_1	reviewer_overall_cnt_2	
# reviewer_overall_cnt_3	reviewer_overall_cnt_4	reviewer_overall_cnt_5	
# reviewer_overall_cnt
class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 iid_voc,
                 cat_voc,
                 brand_voc,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=1):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, iid_voc, cat_voc, brand_voc]:
            self.source_dicts.append(load_dict(source_dict))
        #print(self.source_dicts[0])
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_iid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])
        self.n_brand = len(self.source_dicts[3])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_iid, self.n_cat, self.n_brand

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source= shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)
    def __next__(self):
        return self.next()

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                reviewerID = self.source_dicts[0][ss[8]] if ss[8] in self.source_dicts[0] else 0
                asin = self.source_dicts[1][ss[9]] if ss[9] in self.source_dicts[1] else 0 
                title = ss[15] # title
                brand = self.source_dicts[3][ss[18]] if ss[18] in self.source_dicts[3] else 0
                asin_overall_cnt = ss[19]
                asin_overall_cnt_1 = ss[20] 
                asin_overall_cnt_2 = ss[21]
                asin_overall_cnt_3 = ss[22]
                asin_overall_cnt_4 = ss[23]
                asin_overall_cnt_5 = ss[24]
                reviewer_overall_cnt = ss[25]
                reviewer_overall_cnt_1 = ss[26]
                reviewer_overall_cnt_2 = ss[27]
                reviewer_overall_cnt_3 = ss[28]
                reviewer_overall_cnt_4 = ss[29]
                reviewer_overall_cnt_5 = ss[30]
                label = 1 if int(ss[0]) > 0 else 0
                # asin_hist
                tmp = []
                for fea in ss[1].split("^"):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                asin_hist = tmp
                # brand_hist
                tmp1 = []
                for fea in ss[4].split("^"):
                    c = self.source_dicts[3][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                brand_hist = tmp1
                # top_cat
                tmp2 = []
                for fea in ss[4].split("^"):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp2.append(c)
                top_cat = tmp2
                # category
                tmp = ss[13].split('|')[-1]
                category = self.source_dicts[2][tmp] if tmp in self.source_dicts[2] else 0
                source.append([reviewerId, asin, title, brand, asin_hist, brand_hist, top_cat, category,
                                asin_overall_cnt, asin_overall_cnt_1, asin_overall_cnt_2, asin_overall_cnt_3, 
                                asin_overall_cnt_4, asin_overall_cnt_5, reviewer_overall_cnt, 
                                reviewer_overall_cnt_1, reviewer_overall_cnt_2, reviewer_overall_cnt_3, 
                                reviewer_overall_cnt_4, reviewer_overall_cnt_5])
                target.append([label])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        return source, target
