for file in results/*.csv
do
  python3 plot.py "$file" --save
done
