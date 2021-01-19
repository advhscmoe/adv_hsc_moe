set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'parse_amazon_meta'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;

%DEFAULT input_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_sharded/All_Amazon_Meta.json/par*.gz'

%DEFAULT output_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/All_Amazon_Meta.tsv'

%DEFAULT num_parallel 1000

a = load '$input_file' using PigStorage('\t');

b = foreach a generate flatten(udf.meta2tsv($0)) as (category, asin, tech1, description, fit, 
        title, also_buy, image, tech2, feature, rk, price, 
        also_view, details, main_cat, similar_item, date, brand);

c = filter b by category is not null and TRIM(category) != '';

rmf $output_file
STORE c into '$output_file' using PigStorage('\t', '-schema');
