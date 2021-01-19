#-*- coding:utf-8 -*-
"""Description: Jython UDFs used in Amazon Review tdg pipeline."""
import sys
import copy
import array
import org.apache.pig.data.DataType as DataType
import org.apache.pig.impl.logicalLayer.schema.SchemaUtil as SchemaUtil
import com.xhaus.jyson.JysonCodec as json
import random
reload(sys).setdefaultencoding("utf-8")


@outputSchema('t:tuple(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17)')
def meta2tsv(s):
    try:
        m = json.loads(s.tostring().strip())
        return tuple([str(m[k]).encode('string-escape') for k in ['category', 'asin' , 'tech1', 'description', 'fit',
            'title', 'also_buy', 'image', 'tech2', 'feature', 'rank', 'price',
            'also_view', 'details', 'main_cat', 'similar_item', 'date', 'brand']])
    except:
        return tuple([''] * 18)

# Use schemaFunction to simplify the output schema.
@outputSchema('t:tuple(a0,a1,a2,a3,a4,a5,a6,a7,a8)')
def review2tsv(s):
    try:
        m = json.loads(s.tostring().strip())
        return tuple([str(m[k]) for k in ['overall', 'verified' , 'reviewTime', 'reviewerID', 'asin',
            'reviewerName', 'reviewText', 'summary', 'unixReviewTime']])
    except:
        return tuple([''] * 9)


def convert(s):
    if s is None:
        return ''
    if isinstance(s, array.array):
        s = s.tostring()
    return unicode(s)


def concat_and_clean_category(category):
    cats = [t for t in eval(category.tostring()) if len(t) < 51]
    if len(cats) < 2:
        return ''
    return '|'.join(cats)


@outputSchema('b:bag{t:(%s)}' % ','.join(['a%d' % i for i in range(32)]))
def gen_history(sorted_bag):
    asin_hist, price_hist, overall_hist, brand_hist, ts_hist, category_hist = [], [], [], [], [], []
    out = []
    # for (overall, verified, reviewTime, reviewerID, asin, reviewerName,
    #     reviewText, summary, unixReviewTime, category, tech1, description,
    #     fit, title, also_buy, image, tech2, feature, rank, price, also_view,
    #     details, main_cat, similar_item, date, brand)
    ts = None
    for tmp in sorted_bag:
        tmp = list(tmp)
        new_ts = tmp[8]
        if ts == new_ts:
            continue
        ts = new_ts
        tmp[9] = concat_and_clean_category(tmp[9])
        if not tmp[9]:
            continue
        if len(asin_hist) > 0:
            out.append(tuple(['^'.join(asin_hist), '^'.join(price_hist), '^'.join(overall_hist),
                '^'.join(brand_hist), '^'.join(ts_hist), '^'.join(category_hist)] + tmp))
        asin_hist.append(convert(tmp[4]))
        price_hist.append(convert(tmp[19]))
        overall_hist.append(convert(tmp[0]))
        brand_hist.append(convert(tmp[-1]))
        ts_hist.append(convert(tmp[8]))
        category_hist.append(tmp[9])
    return out


@outputSchema('b:boolean')
def HasMoreThanOneCategoryLevel(s):
    try:
        return len(eval(s.tostring())) > 1
    except:
        return False


@outputSchema('b:bag{t:(a)}')
def ExtractCategories(s):
    return [str(t) for t in eval(s.tostring())]


@outputSchema('t:tuple(a1,a2,a3,a4,a5)')
def TallyOverall(overall_bag):
    cnts = [0, 0, 0, 0, 0]
    for o in overall_bag:
        cnts[int(float(o[0].tostring())) - 1] += 1
    return tuple(cnts)


def gen_pair(prev_list, curr_list):
    return tuple(prev_list[:4] + [prev_list[5], prev_list[4]] +
        prev_list[-6:] + ([curr_list[4]] + curr_list[-6:]))


# asin_hist, price_hist, overall_hist, brand_hist, reviewerID, overall_a, asin_a,
# reviewText_a, categories_a, title_a, price_a, brand_a, verall_b, asin_b,
# reviewText_b, categories_b, title_b, price_b, brand_b
@outputSchema('b:bag{t:(%s)}' % ','.join(['a%d' % i for i in range(19)]))
def gen_pairwise(pointwise_bag):
    pointwise_bag.sort(key=lambda t: t[0].count('^'))
    overall = None
    previous_tuple = None
    # Let's only generate near-consecutive nontrivial pairs to ensure
    # user histories are as close as possible.
    out = []
    for t in pointwise_bag:
        if overall is None:
            overall = t[4]
            previous_tuple = t
            continue
        if overall is not None and overall != t[4]:
            out.append(gen_pair(list(previous_tuple), list(t)))
            overall = t[4]
            previous_tuple = t
    return out


@outputSchema('t:tuple(a, b)')
def top_categories(category):
    categories = convert(category).split('|')
    if len(categories) == 0:
        return tuple(['', ''])
    elif len(categories) == 1:
        return tuple([categories[0], ''])
    return tuple(categories[:2])


@outputSchema('b:bag{t:(%s)}' % ','.join(['a%d' % i for i in range(40)]))
def gen_negatives(pointwise_bag):
    random.shuffle(pointwise_bag)
    out = []
    prev_t = None
    for t in pointwise_bag:
        # (top_cat_size,
        # asin_hist, price_hist, overall_hist, brand_hist,
        # unixReviewTime_hist, category_hist, overall, reviewerID, asin, reviewText,
        # summary, unixReviewTime, category, description, title, also_buy, feature,
        # rank, price, also_view, details, top_cat, next_cat,
        # main_cat, similar_item, brand, asin_overall_cnt_1,
        # asin_overall_cnt_2,
        # asin_overall_cnt_3, asin_overall_cnt_4, asin_overall_cnt_5, asin_overall_cnt,
        # reviewer_overall_cnt_1,
        # reviewer_overall_cnt_2, reviewer_overall_cnt_3, reviewer_overall_cnt_4,
        # reviewer_overall_cnt_5,  reviewer_overall_cnt, rand) = t
        if prev_t:
            out.append(prev_t)
            out[-1][0] = '-' + convert(t[0])   # negative label
            out[-1][1:7] = t[1:7]   # xxx_hist
            out[-1][8] = t[8]   # reviewerID
            out[-1][-7:] = t[-7:]   # reviewer_overall_cnt_x
            out[-1] = tuple(out[-1])
        prev_t = list(t)
    return out


@outputSchema('b:bag{t:(level,cat)}')
def split_category(category, prefix=0):
    categories = convert(category).split('|')
    if prefix:
        categories = ['|'.join(categories[:i]) for i in range(1, len(categories))]
    return list(enumerate(categories))
