# Age Gender Estimation (Tensorflow)

*Tensorflow based Age Gender Model (CPU or GPU Only)*

Only needs ~200MB of GPU memory (more will not make it faster)

| Hardware 	| Inference Time (Milliseconds)
|----------	|-------------------------------
| CPU      	| 700ms per person
| GPU       | 430 ms total

This model's inference time increases sub-linearly with the number of people.

| Detection       | Accuracy
|---------------	|-------------------------------
| Face - Accurate | 92% (frontal faces) (accurate model)
| Face - Fast     | 88% (frontal faces) (accurate model)
| Age      	      | +- 5 years

There are two detectors built into this container.

| Model     | Description
|----------	|-------------------------------
| Fast      | dlib 98.38% on the Labeled Faces in the Wild benchmark
| Accurate  | tensorflow cnn face detector 99.38% on the Labeled Faces in the Wild benchmark

- The size of the face is 64x64
- Tensorflow has "allow growth" mode on, so it will dynamically allocate GPU memory.

## Run
```sh
docker run -ti \\
-p 8080:8080 \\
sugarkubes/tensorflow-age-gender:cpu
```


## Routes

`GET /`
`GET /health`
`GET /healthz`
- Respond with a 200 for healthcheck

`POST /predict`
- Example:
```sh
curl -X POST \\
http://0.0.0.0:8080/predict \\
-H 'Content-Type: application/json' \\
-H 'Authorization: Basic c3VnYXI6a3ViZXM=' \\
-d '{ "get_attributes": true, "url": "https://s3.us-west-1.wasabisys.com/public.sugarkubes/repos/sugar-cv/object-detection/friends.jpg" }'
```

- Post parameters
```json
{
  "confidence": 0.5,
  "nms": 0.5, # non-max supression
  "draw": true,
  "face_detector": "fast", # One of ["accurate", "fast"]
  "get_attributes": true,
  "return_image": true, # use false for production/faster results
  "url": 'https://your-image.jpg', # use url or b64 image
  "b64": "", # base 64 encoded image
}
```


## ENV Variables

| Variable 	   | Default
|------------  |-------------------------------
| PORT         | 8080
| GPU          | None (True for GPU version)
| GPU_FRACTION | 0.25 (25% of the gpu will be allocated to this model)


## Configs
Located at `/age-gender-estimateion/config.json`
```bash
{
  "host": "0.0.0.0",
  "port": 8080,
  "log": false,
  "basic_auth_username": "sugar",
  "basic_auth_password": "kubes"
}
```

## Google Cloud Run Enabled

- This container can immediately be deployed using Google's new Cloud Run serverless service (cpu inference only)
- Its a cheap and quick way to get the model online

## Response

```json
{
  "confidence": 0.5,
  //         x1   y1   x2   y2   w    h  conf age gender
  "faces": [[451, 0, 914, 452, 463, 514, -1, 37, "M"]],
  "image_size": [1920, 1080],
  "inference_time": 1363.462,
}
```



## About the container
- python 3.5.2 (cpu)
- python 3.5.1 (gpu)
- tensorflow 1.13.1
-

## validated hosts
- intel x86 machines running ubuntu server
- cuda 10.1
- cuda 10.2
