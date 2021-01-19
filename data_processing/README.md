## Pig scripts to process amazon public dataset.

1. split single gz files into 1000 shards for faster mapreduce processing
amazon_{meta,csv,review}_split.pig

2. parse review and meta files from json to tsv:
parse_amazon_{review,meta}.pig

3. join meta and review data (csv is the core file, which is not really needed)
join_all.pig

4. aggregate_user.pig get user review history

5. trim useless columns and remove long category names
clean_data.pig

6. aggregate review counts for each grade, along both item and reviewer dimension.
aggregate_items.pig  
