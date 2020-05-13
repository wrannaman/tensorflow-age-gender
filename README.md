# Age Gender Estimation (Tensorflow)

*Tensorflow based Age Gender Model (CPU + GPU)*

## Speed
| Hardware 	| Inference Time (Milliseconds)
|----------	|-------------------------------
| CPU      	| 750 (intel i-7)
| GPU      	| 430 (NVIDIA 2080)

This model's inference time increases sublinearly with the number of people.

## Accuracy
| Detection       | Accuracy
|---------------	|-------------------------------
| Face - Accurate | ~95% (frontal faces) (accurate model)
| Face - Fast     | ~90% (frontal faces) (accurate model)
| Age      	      | +- 5 years

There are two detectors built into this container. You can toggle between them in the post parameters

## Face Detector

| Model     | Description
|----------	|-------------------------------
| Fast      | dlib
| Accurate  | tensorflow based cnn face detector

- The size of the face is 64x64
- For the GPU version, this model needs a minimum of 200MB.

## Run
```sh

#cpu
docker run -ti \\
-p 9090:9090 \\
sugarkubes/tensorflow-age-gender:cpu

#gpu
nvidia-docker run -ti \\
-p 9090:9090 \\
sugarkubes/tensorflow-age-gender:gpu
```


## Routes

`GET /`
`GET /health`
`GET /healthz`
- Responds with a 200 for healthcheck

`POST /predict`
- Example:
```sh
curl -X POST \\
http://0.0.0.0:9090/predict \\
-H 'Content-Type: application/json' \\
-d '{ "url": "https://s3.us-west-1.wasabisys.com/public.sugarkubes/repos/sugar-cv/object-detection/friends.jpg" }'
```

- Post parameters
```json
{
  "face_detector": "fast", # One of ["accurate", "fast"]
  "return_image": true, # use false for production/faster results
  "url": 'https://your-image.jpg', # use url or b64 image
  "b64": "", # base 64 encoded image
}
```


## ENV Variables

| Variable 	   | Default
|------------  |-------------------------------
| PORT         | 8080
| HOST         | 0.0.0.0
| GPU          | "" (true for GPU version)
| GPU_FRACTION | 0.25 (25% of the gpu will be allocated to this model)
| BASIC_AUTH_USERNAME | ""
| BASIC_AUTH_PASSWORD | ""

## Google Cloud Run Enabled

- This container can immediately be deployed using Google's new Cloud Run serverless service (cpu inference only)
- Its a cheap and quick way to get the model online

## Response

```json
{
  //         x1   y1   x2   y2   w    h  conf age gender
  "faces": [[451, 0, 914, 452, 463, 514, -1, 37, "M"]],
  "image_size": [1920, 1080],
  "inference_time": 431.462,
}
```


## Authentication
- The container comes with the ability to support basic auth.
- basic auth is disabled by default.
- to turn it off, set `BASIC_AUTH_USERNAME=""` and `BASIC_AUTH_PASSWORD=""`
- to turn it on, set `BASIC_AUTH_USERNAME="root"` and `BASIC_AUTH_PASSWORD="your password"`

## About the container
- python 3.5.2 (cpu)
- python 3.5.1 (gpu)
- tensorflow 1.13.1

## Validated Hosts
- intel x86 machines running ubuntu server 18.04 19.04 20.04
- cuda 10.1
- cuda 10.2
- mac
