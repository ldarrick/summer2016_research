#!/bin/bash

cat $1 | cut -d, -f1,2,9,10,11,13,17,23,25 | grep ^$2, >> $3
