#!/bin/bash

# call this script where the synthetic folder is, with a copy of pamonoDataToYolo.py in every folder (1, 2, and 3)

# 1,2,3? !!!
number="3"

# image dimensions !!!
width="450"
height="170"

# where to move all images and annotations !!!
#1
#dest_dir="/home/mik/ma/DarknetAlexey/darknet/data/100nmONE/train"
#2
#dest_dir="/home/mik/ma/DarknetAlexey/darknet/data/100nmONE/valid"
#3
dest_dir="/home/mik/ma/DarknetAlexey/darknet/data/100nmONE/test"

# darknet root
dr=/home/mik/ma/DarknetAlexey/darknet

csv="NanoSynthMLPolygonFormFactors.csv"
prefix="imagesAndAnnotations"

########

rm synthetic/${number}/*png
rm -r synthetic/${number}/background_component

cp pamonoDataToYolo.py synthetic/${number}/particles_component/pamonoDataToYolo.py
cp synthetic/${number}/${csv} synthetic/${number}/particles_component/${csv}
cd synthetic/${number}/particles_component
#rm *png
#rm -r background_component/
python3 pamonoDataToYolo.py --csvFileName=${csv} --imageWidth=${width} --imageHeight=${height} --prefix=${prefix}
cd ${prefix}
mv * ${dest_dir}
cd ${dr}

# !!!
#1
#ls -1 data/100nm/train/*.png > data/100nmONE/train/train.txt
#2
#ls -1 data/100nm/valid/*.png > data/100nmONE/valid/valid.txt
#3
ls -1 data/100nm/test/*.png > data/100nmONE/test/test.txt
