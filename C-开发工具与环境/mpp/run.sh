docker stop -t 0 ai
docker rm ai
docker run --name ai -d \
       -p 8000:9999 \
       -v $PWD/training_notebook:/opt/notebook \
       caapi/ai