#! /bin/bash

curl https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=1 -o model-r100-arcface-ms1m-refine-v2.zip -J -L -k
unzip model-r100-arcface-ms1m-refine-v2.zip -d src/api/models
rm model-r100-arcface-ms1m-refine-v2.zip
