#!/bin/bash

set -e

# I found this here https://www.kaggle.com/datasets/aydosphd/webscreenshots
mkdir -p resources/website_screenshots
cd resources/website_screenshots
gdown --folder https://drive.google.com/drive/folders/1cNjcwM0rc8Wr9dESrsxytKwXl9j3QnbR
7za x screenshots-1440x900/machinery.7z
7za x screenshots-1440x900/music.7z
7za x screenshots-1440x900/sport.7z
7za x screenshots-1440x900/tourism.7z
rm -rf screenshots-1440x900
