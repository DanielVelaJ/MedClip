#!/usr/bin/bash
cd ../data/raw
echo "Starting download, takes around a day and a half lol"
wget -r -N -c -np --user danielvelaj --password Lucicaroro1! https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz

wget -r -N -c -np --user danielvelaj --password Lucicaroro1! https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip

wget -r -N -c -np --user danielvelaj --password Lucicaroro1! https://physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv.gz

wget -r -N -c -np --user danielvelaj --password Lucicaroro1! https://physionet.org/files/mimic-cxr-jpg/2.0.0/


# Now unzip gzip files
# First go to directory
echo "Unzipping everything"
cd ./physionet.org/files/mimic-cxr/2.0.0

gzip -d *.gz
unzip *.zip

# Repeat for the other directory
cd ../../../files/mimic-cxr-jpg/2.0.0

gzip -d *.gz
unzip *.zip

