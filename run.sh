python3 src/make_paper_figures.py --peels 1 $1

inputfile=$(echo $1 | cut -d '/' -f 2)

cd paper_figures/
for i in tri*.pdf his*.pdf data*.csv hist*.csv
do
    cp $i $inputfile"_"$i
done
cd ../
