set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'remove_dirty_columns'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;

%DEFAULT base_input '$HDFS_ROOT/amazon_data_processed/user_aggregated.with_negatives'
%DEFAULT input_file '$base_input.train.tsv'
%DEFAULT input_file2 '$base_input.test.tsv'

%DEFAULT base_output '$HDFS_ROOT/amazon_data_processed/remove_dirty'
%DEFAULT output_file '$base_output.train.tsv'
%DEFAULT output_file2 '$base_output.test.tsv'


%DEFAULT num_parallel 1000

train = load '$input_file' using PigStorage('\t', '-schema');

loaded = foreach train generate top_cat_size,
        asin_hist, price_hist, overall_hist, brand_hist,
        unixReviewTime_hist, category_hist, overall, reviewerID, asin, reviewText,
        summary, unixReviewTime, category, description, title,
        price, top_cat, next_cat,
        brand, asin_overall_cnt_1,
        asin_overall_cnt_2,
        asin_overall_cnt_3, asin_overall_cnt_4, asin_overall_cnt_5, asin_overall_cnt,
        reviewer_overall_cnt_1,
        reviewer_overall_cnt_2, reviewer_overall_cnt_3, reviewer_overall_cnt_4,
        reviewer_overall_cnt_5,  reviewer_overall_cnt;

rmf $output_file
STORE loaded into '$output_file' using PigStorage('\t', '-schema');

test = load '$input_file2' using PigStorage('\t', '-schema');

loaded2 = foreach test generate top_cat_size,
        asin_hist, price_hist, overall_hist, brand_hist,
        unixReviewTime_hist, category_hist, overall, reviewerID, asin, reviewText,
        summary, unixReviewTime, category, description, title,
        price, top_cat, next_cat,
        brand, asin_overall_cnt_1,
        asin_overall_cnt_2,
        asin_overall_cnt_3, asin_overall_cnt_4, asin_overall_cnt_5, asin_overall_cnt,
        reviewer_overall_cnt_1,
        reviewer_overall_cnt_2, reviewer_overall_cnt_3, reviewer_overall_cnt_4,
        reviewer_overall_cnt_5, reviewer_overall_cnt;

rmf $output_file2
STORE loaded2 into '$output_file2' using PigStorage('\t', '-schema');