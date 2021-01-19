set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'gen_pairwise'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;


%DEFAULT input_file '$HDFS_ROOT/amazon_data_processed/user_aggregated.item_aggregated.tsv'
%DEFAULT output_file '$HDFS_ROOT/amazon_data_processed/pairwise.trimmed.tsv'

%DEFAULT num_parallel 300

/*
%DEFAULT output_file 'w'
%DEFAULT input_file 'y'
%DEFAULT num_parallel 1
*/

loaded = load '$input_file' using PigStorage('\t', '-schema');

trimmed = foreach loaded generate asin_hist, price_hist, overall_hist, brand_hist,
        overall, reviewerID, asin, reviewText, categories, title, price, brand;

grouped = foreach (group trimmed by reviewerID parallel $num_parallel) {
        generate flatten(udf.gen_pairwise(trimmed)) as (asin_hist, price_hist, overall_hist, brand_hist,
        reviewerID, overall_a, asin_a, reviewText_a, categories_a, title_a, price_a, brand_a,
        overall_b, asin_b, reviewText_b, categories_b, title_b, price_b, brand_b);
};

rmf $output_file
STORE grouped into '$output_file' using PigStorage('\t', '-schema');