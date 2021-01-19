set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'aggregate_user'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;


%DEFAULT input_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/joined.tsv'
%DEFAULT output_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/user_aggregated.new.tsv'
%DEFAULT num_parallel 1000


/*
%DEFAULT input_file 'z'
%DEFAULT output_file 'y'
%DEFAULT num_parallel 1
*/

joined = load '$input_file' using PigStorage('\t', '-schema');
filtered = filter joined by udf.HasMoreThanOneCategoryLevel(category);
a = foreach (group filtered by reviewerID parallel $num_parallel ) {
    sorted = order filtered by unixReviewTime ASC;
    generate flatten(udf.gen_history(sorted)) as (
        asin_hist, price_hist, overall_hist, brand_hist, unixReviewTime_hist, category_hist,
        overall, verified, reviewTime, reviewerID, asin, reviewerName, reviewText, summary, unixReviewTime, category,
        tech1, description, fit, title, also_buy, image, tech2, feature, rank, price, also_view, details, main_cat,
        similar_item, date, brand);
};

rmf $output_file
STORE a into '$output_file' using PigStorage('\t', '-schema');