#!/bin/bash
set -x

mkdir -p $1

(cd $1 && wget http://www.bio8.cs.hku.hk/RENET2/renet2_data_models.tar.gz && tar -xf renet2_data_models.tar.gz && rm renet2_data_models.tar.gz )
