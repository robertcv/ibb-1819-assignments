#!/bin/bash

find pos_left/ -name '*.jpg' -exec echo {} 1 0 0 100 100 \; > pos_left.dat
find pos_right/ -name '*.jpg' -exec echo {} 1 0 0 100 100 \; > pos_right.dat
find neg/ -name '*.jpg' > neg.dat

opencv_createsamples -info pos_left.dat -w 24 -h 48 -vec ear_left.vec -num 336
opencv_createsamples -info pos_right.dat -w 24 -h 48 -vec ear_right.vec -num 256

opencv_traincascade -data . -vec ear_left.vec -bg neg.dat -numStages 15 -numPos 230 -numNeg 1000 -minHitRate 0.95 -w 24 -h 48 -precalcValBufSize 4000 -precalcIdxBufSize 4000
mv cascade.xml cascade_left_ear.xml
rm stage*.xml
rm params.xml

opencv_traincascade -data . -vec ear_right.vec -bg neg.dat -numStages 15 -numPos 170 -numNeg 1000 -minHitRate 0.95 -w 24 -h 48 -precalcValBufSize 4000 -precalcIdxBufSize 4000
mv cascade.xml cascade_right_ear.xml
rm stage*.xml
rm params.xml
