#!/bin/bash
source $1/bin/activate
export PYTHONPATH=~/repos/pythia
for i in $(seq 1 $2)
do 
  echo "Running process $i"
  python trainer/fpm_backtest.py
  python trainer/back_up_best_model.py data/seed.txt data/models/fpm data/models/bkk
done
