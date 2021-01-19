set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 107374182
set pig.minCombinedSplitSize 107374182
set mapreduce.map.memory.mb 5000
set mapreduce.reduce.memory.mb 8000
-- set mapreduce.map.java.opts -Xmx2764m
-- set mapreduce.reduce.java.opts -Xmx2764m
-- set mapreduce.task.io.sort.mb 2000
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'split_amazon_review'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
register 'amazon_data_udf.py' using jython AS udf;

%DEFAULT input_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data/all_csv_files.csv'

%DEFAULT output_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_sharded/all_csv_files.tsv'

%DEFAULT num_parallel 1000

a = load '$input_file' using PigStorage(',');

b = foreach (group a by $1 parallel $num_parallel) generate flatten(a) as reviewerID, asin, rating, timestamp;

rmf $output_file
STORE b into '$output_file' using PigStorage('\t', '-schema');
