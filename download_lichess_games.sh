#!/bin/bash


set -e

mkdir -p resources/lichess_games
cd resources/lichess_games
wget "https://database.lichess.org/standard/lichess_db_standard_rated_2013-04.pgn.zst"
wget "https://database.lichess.org/standard/lichess_db_standard_rated_2013-05.pgn.zst"

pzstd --rm -d lichess_db_standard_rated_2013-04.pgn.zst
pzstd --rm -d lichess_db_standard_rated_2013-05.pgn.zst