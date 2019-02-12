#!/bin/bash
# ref: https://jefflirion.github.io/sphinx-github-pages.html
PWD=`pwd`
if [ `basename $PWD` != "hsnf" ]; then
  echo "please move to root dir"
  exit 1
fi

mv ./docs ./html
cd ./html
make html
cd -
mv ./html ./docs
