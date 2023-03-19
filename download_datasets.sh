#!/bin/bash

[[ -z "$DATA_DIR" ]] && { echo "Please set DATA_DIR to something" ; exit 1; }

BBEXG_DATA_DIR=$DATA_DIR/bbexg
mkdir -p $BBEXG_DATA_DIR
cd $BBEXG_DATA_DIR

# social media
wget http://konect.cc/files/download.tsv.youtube-u-growth.tar.bz2
wget https://nrvis.com/download/data/soc/soc-FourSquare.zip
wget https://nrvis.com/download/data/soc/soc-digg.zip
wget http://snap.stanford.edu/data/loc-gowalla_edges.txt.gz 

# others
wget http://snap.stanford.edu/data/as-skitter.txt.gz
wget http://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz

# extract
gunzip  *.gz
tar -xf *.bz2
unzip -o soc-FourSquare.zip
unzip -o soc-digg.zip

cd -