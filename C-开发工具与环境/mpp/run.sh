docker stop -t 0 ai
docker rm ai
docker run --name ai -d \
       -p 5001:5000 \
       -p 8000:9999 \
       -p 6380:6379 \
       -v $PWD/app:/data/app \
       -v $PWD/notebook:/opt/notebook \
       caapi/ai