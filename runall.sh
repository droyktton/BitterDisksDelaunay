#procesamos todos los .dat en carpeta data/
for i in data/*.dat; do bash run.sh $i; done

#convertirmos todos los pdf en paper_figures/ a eps

cd paper_figures/
for i in *.pdf; do pdftops -eps "$i" "${i%.pdf}.eps"; done
cd ../
