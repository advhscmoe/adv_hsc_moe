set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'parse_amazon_review'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;

%DEFAULT input_file '$HDFS_ROOT/amazon_data_sharded/All_Amazon_Review.json/par*.gz'

%DEFAULT output_file '$HDFS_ROOT/amazon_data_processed/All_Amazon_Review.tsv'

%DEFAULT num_parallel 1000

a = load '$input_file' using PigStorage('\t');

b = foreach a generate flatten(udf.review2tsv($0)) as (overall, verified, reviewTime, reviewerID, asin, 
            reviewerName, reviewText, summary, unixReviewTime);

c = filter b by unixReviewTime is not null and TRIM(unixReviewTime) != '';

rmf $output_file
STORE c into '$output_file' using PigStorage('\t', '-schema');
