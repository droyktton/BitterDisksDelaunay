awk -F',' 'NR>1 {sum += $4} END {print sum}' paper_figures/data.csv
