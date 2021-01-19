set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'agg_cat'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;


%DEFAULT input_file '$HDFS_ROOT/amazon_data_processed/user_aggregated.tsv'
%DEFAULT output_file '$HDFS_ROOT/amazon_data_processed/category_count'

%DEFAULT num_parallel 100

/*
%DEFAULT input_file 'z'
%DEFAULT output_file 'y'
%DEFAULT num_parallel 1
*/

loaded = load '$input_file' using PigStorage('\t', '-schema');
flattened = foreach loaded generate flatten(udf.ExtractCategories(category)) as category;
counted = foreach (group flattened by category parallel $num_parallel ) generate group as category, COUNT(flattened) as category_cnt;
filtered = filter counted by category_cnt > 1;
sorted = order filtered by category_cnt DESC;
rmf $output_file
STORE sorted into '$output_file' using PigStorage('\t', '-schema');