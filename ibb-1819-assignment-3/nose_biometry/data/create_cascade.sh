#!/bin/bash

opencv_createsamples -info pos.dat -w 24 -h 24 -vec pos.vec -num 350

opencv_traincascade -data . -vec pos.vec -bg neg.dat -numStages 15 -numPos 170 -numNeg 1000 -minHitRate 0.95 -w 24 -h 24
mv cascade.xml my_nose_cascade.xml
rm stage*.xml
rm params.xml