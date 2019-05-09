# Age Gender Estimation (Tensorflow)

## Build docker
```sh
docker build \
-t registry.sugarkubes.io/sugar-cv/tensorflow-age-gender:cpu \
-f Dockerfile.cpu .
```

## Run
```sh
docker run -ti \
-p 9090:9090 \
registry.sugarkubes.io/sugar-cv/tensorflow-age-gender:cpu
```
## Call

```sh
curl -X POST   http://0.0.0.0:9090/predict   -H 'Content-Type: application/json'   -H 'Authorization: Basic c3VnYXI6a3ViZXM='   -d '{ "get_attributes": true, "url": "https://s3.us-west-1.wasabisys.com/public.sugarkubes/repos/sugar-cv/object-detection/friends.jpg" }' 
```

## Google Cloud Run Enabled.

- This container can immediately be deployed using Google's new Cloud Run service (cpu inference only)

## Request

- Post request to /predict

```json
{
  "confidence": 0.5,
  "draw": true,
  "get_attributes": true,
  "return_image": true,
  "face_detector": 'accurate', # one of ['fast', 'accurate']
}
```

## Response

```json
{
  "confidence": 0.5,
  "face_detector": "fast",
  //         x1    y1   x2   y2   w    h  conf age gender
  "faces": [[451, -62, 914, 452, 463, 514, -1, 37, "M"]],
  "image_size": [1920, 1080],
  "inference_time": 1363.462,
}
```
