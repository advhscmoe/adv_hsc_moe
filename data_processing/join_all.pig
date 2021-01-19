set mapred.output.compress true
set pig.splitCombination true
set pig.maxCombinedSplitSize 1073741824
set pig.minCombinedSplitSize 1073741824
set mapreduce.job.reduce.slowstart.completedmaps 1.0
set mapred.reduce.tasks.speculative.execution true
set mapred.map.tasks.speculative.execution true
set job.name 'join_all'

SET output.compression.enabled true;
SET output.compression.codec org.apache.hadoop.io.compress.GzipCodec;
REGISTER 'jyson-1.0.2/lib/jyson-1.0.2.jar';
REGISTER 'amazon_data_udf.py' using jython AS udf;

%DEFAULT meta_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/All_Amazon_Meta.tsv/par*.gz'
%DEFAULT review_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/All_Amazon_Review.tsv/par*.gz'
-- %DEFAULT csv_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_sharded/all_csv_files.tsv/par*.gz'
%DEFAULT output_file '/user/recsys/rank_dev/yunjiang.jiang/amazon_data_processed/joined.tsv' 
meta = load '$meta_file' using PigStorage('\t', '-schema');
review = load '$review_file' using PigStorage('\t', '-schema');
-- csv = load '$csv_file' using PigStorage('\t', '-schema'); 

-- b = join review by (reviewerID, asin, overall, unixReviewTime), csv by (reviewerID, asin, rating, timestamp);
-- b = foreach b generate $0 as overall, $1 as verified, $2 as reviewTime, $3 as reviewerID, $4 as asin, 
--     $5 as reviewerName, $6 as reviewText, $7 as summary, $8 as unixReviewTime;
c = join review by asin, meta by asin parallel 1000;

c = foreach c generate $0 as overall, $1 as verified, $2 as reviewTime, $3 as reviewerID, $4 as asin, 
    $5 as reviewerName, $6 as reviewText, $7 as summary, $8 as unixReviewTime, $9 as category,
    $11 as tech1, $12 as description, $13 as fit, $14 as title, $15 as also_buy, $16 as image, 
    $17 as tech2, $18 as feature, $19 as rank, $20 as price, $21 as also_view, $22 as details, 
    $23 as main_cat, $24 as similar_item, $25 as date, $26 as brand;


rmf $output_file
STORE c into '$output_file' using PigStorage('\t', '-schema');