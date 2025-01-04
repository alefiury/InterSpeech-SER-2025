# !/bin/bash

# Download rir dataset
wget -c "https://us.openslr.org/resources/28/rirs_noises.zip"

# Check if the directory "augmentation_data" exists, if not, create it
if [ ! -d "augmentation_data" ]; then
  mkdir augmentation_data
fi

# Extract rirs_noises into augmentation_data
unzip rirs_noises.zip -d augmentation_data

# Download musan dataset
wget -c "https://us.openslr.org/resources/17/musan.tar.gz"

# extract musan into augmentation_data
tar -xvzf musan.tar.gz -C augmentation_data

# remove directory "speech" inside musan
rm -rf augmentation_data/musan/speech

rm rirs_noises.zip
rm musan.tar.gz