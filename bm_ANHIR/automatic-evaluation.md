# Automatic evaluation

This is short description with instruction notes how to create image to be upload to [grand-challenge.org](https://www.grand-challenge.org).
For newbies, please see [Get started with Docker](https://docs.docker.com/get-started).
First you need to install Docker.io and run everything as super user `sudo -i`.


## Build

Copy the required data to the local directory

```bash
docker build -t anhir -f Dockerfile .
```

## Run and Test

Running the docker image with mapped folders 
```bash
mkdir submission output
```
and upload the sample submission to `submission` and run the image
```bash
docker run --rm -it \
        --memory=4g \
        -v $(pwd)/submission/:/input/ \
        -v $(pwd)/output/:/output/ \
        anhir
```


## Export 

Export the created image to be uploaded to the evaluation system.
```bash
docker save anhir > anhir.tar } | gzip
```

## Browsing images

**Browsing**
To see your local biulded images use:
```bash
docker image ls
```

**Cleaning**
In case you fail with some builds, you may need to clean your local storage.
```bash
docker image prune
```
or [Docker - How to cleanup (unused) resources](https://gist.github.com/bastman/5b57ddb3c11942094f8d0a97d461b430)
```bash
docker images | grep "none"
docker rmi $(docker images | grep "none" | awk '/ / { print $3 }')
```


## References

* https://evalutils.readthedocs.io/en/latest/usage.html
* https://grand-challengeorg.readthedocs.io/en/latest/evaluation.html
