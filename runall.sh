#procesamos todos los .dat en carpeta data/
for i in data/*.dat; do bash run.sh $i; done


cd paper_figures/

# convertimos todos los pdf a eps
for i in *.pdf; do pdftops -eps "$i" "${i%.pdf}.eps"; done

# convertimos todos los pdf a jpg
for f in *.pdf; do pdftoppm -jpeg -singlefile -r 300 "$f" "${f%.pdf}"; done

cd ../

