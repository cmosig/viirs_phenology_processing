#! /bin/bash

username="cmosig"

wget -r -np -R "index.html*" -R "*.jpg" --recursive --no-parent --user $username --ask-password https://e4ftl01.cr.usgs.gov/VIIRS/VNP22Q2.001/
