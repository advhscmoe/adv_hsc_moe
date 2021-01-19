set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'count_categories'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
-- This is awful UDF. Don't waste time on it!
-- register '../gen_trainset/datafu-pig-1.5.0.jar';
-- define Enumerate datafu.pig.bags.Enumerate('1');
REGISTER 'amazon_data_udf.py' using jython AS udf;
/*
%DEFAULT base_output '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/gnn_codist_base'
%DEFAULT input_file '$base_output.train.tsv'
%DEFAULT output_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/category_counts.train'
%DEFAULT num_parallel 100
*/

%DEFAULT input_file 'clean'
%DEFAULT output_file 'count'
%DEFAULT num_parallel 1

train = load '$input_file' using PigStorage('\t', '-schema');

flattened = foreach train generate flatten(udf.split_category(category, 1)) as (level, cate);

counted = foreach (group flattened by (level, cate) parallel $num_parallel) generate flatten(group) as (level, cate), COUNT(flattened) as cnt;

rmf $output_file
STORE counted into '$output_file' using PigStorage('\t', '-schema');