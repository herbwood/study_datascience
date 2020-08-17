#!/bin/bash

pip install face-recognition

mkdir data
cd data
wget -O dami.jpg https://mn.kbs.co.kr/data/news/2018/07/25/4014554_7jN.jpg
wget -O many.jpg https://nkcf.org/wp-content/uploads/2017/11/people.jpg
wget -O obama.jpg https://github.com/ageitgey/face_recognition/blob/master/examples/obama.jpg?raw=true
wget -O biden.jpg https://github.com/ageitgey/face_recognition/blob/master/examples/biden.jpg?raw=true
wget -O ohmygirl.jpg https://v-phinf.pstatic.net/20200529_158/1590718251214YxgC7_JPEG/upload_720X535_VEC95B1EC8DB8EB84A4EC9DBC.jpg?type=f886_499
cd ../
