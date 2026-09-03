python3 src/make_paper_figures.py --peels 1 $1

inputfile=$(echo data/T00001_D30B16.dat | cut -d '/' -f 2)

cd paper_figures/
for i in *.pdf *.csv
do
    cp $i $inputfile"_"$i
done
cd ../