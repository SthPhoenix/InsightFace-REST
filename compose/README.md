# Running InsightFace-REST with docker-compose

## Availiable options:
There are multiple docker-compose files provided:
1. `docker-compose.yml` - start single container using 1 GPU 
2. `docker-compose-cpu.yml` - start single container using CPU
3. `docker-compose-multi-gpu.yml` - start 2 IFR containers on 2 GPUs and add NGINX container as load balancer
4. `docker-compose-v2.yml` - provide all options above using docker-compose v2 syntax


### Docker-compose v1

If you are using docker-compose v1 you can run desired variant by
executing following command in console:

```docker-compose -f {compose_file_name} up```

For example to run CPU version you can use:

   ```docker-compose -f docker-compose-cpu.yml up```

Or to use 2 `GPUs`*:

   ```docker-compose-multi-gpu.yml```

> *Since load balancer is used for accessing containers on different GPUs,
default port changed from `18081` to `18080`

### Docker-compose v2

Docker compose v2 provides `profiles` option which can be used to specify containers
which should be run.

To run `1` container on `1` GPU execute**:
```
docker compose -f docker-compose-v2.yml --profile gpu up --build
```

To run `2` containers on `2` GPUs execute* **:
```
docker compose -f docker-compose-v2.yml --profile mgpu up --build
```

To run `1` container on CPU execute**:
```
docker compose -f docker-compose-v2.yml --profile cpu up --build
```


> *Since load balancer is used for accessing containers on different GPUs,
default port changed from `18081` to `18080`
> 
> **Notice the `docker compose ...` instead of `docker-compose ...` - it's not a mistake,
docker-compose v2 is now docker plugin and should be executed this way.


## Running on multiple GPUs

When you run InsightFace-REST on multiple GPUs docker-compose creates one container per GPU,
which might lead to race conditions when starting for the first time. 

On first run models are downloaded and converted to TensorRT engines, and if you run multiple containers at once
every container will try to download and convert model.

To solve this problem just start stack using one gpu, let it download and convert everything and then start stack 
for multiple GPUs.

Also keep in mind that TRT engines are locked to exact GPU model, so in case you are using different models of
GPUs on single system you should specify different model dirs for different GPUs to ensure they use correct TRT `.plan`
files.

## Changing default configuration

All configs  related to detection and recognition models, num workers, etc., are located in `.env` file. 

For meaning of each parameter you can refer to [deploy_trt.sh](../deploy_trt.sh) script.

As starting point it's recommended to adjust `NUM_WORKERS` parameter to value which fits to your GPU
model without causing CUDA Out of Memory errors (on modern GPUs usually about 4-6 workers can fit in memory and achieve 
maximum performance, larger number of workers usually gives no performance boost)

