# Running Server

```bash
docker run --gpus all --rm -it -v \
/path/to/sandwich-classifier/server/model:/tmp/mounted_model/0001 -p 8501:8501 \
-t gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0-gpu:latest
```

Run the following in another terminal
`python3 main.py`
