#!/bin/bash

for file in *.eps
do
  f=${file%.*}
  convert -density 1024 $file -resize 1024x1024 $f.jpeg
done
