#!/bin/bash
#this is a bash file to make scraping go faster
echo "Running iteration..."
#number of times to repeat and seconds in between when pulls error and stops
num_repeats=50
wait_time=5

for ((i=1; i<=num_repeats; i++))
do
    echo "Running iteration $i..."
    python popular_scrape.py 
    echo "Waiting for $wait_time seconds..."
    sleep $wait_time
done

echo "Finished running the script $num_repeats times."
