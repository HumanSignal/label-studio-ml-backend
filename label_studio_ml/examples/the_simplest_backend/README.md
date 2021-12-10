## Quickstart

Build and start Machine Learning (ML) backend on `http://localhost:9090`. 

From the command line, run the following:
```bash
docker-compose up
```

Then, check the status of the ML backend:
```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

After starting the ML backend, connect it to Label Studio. See [Add an ML backend to Label Studio](https://labelstud.io/guide/ml.html#Add-an-ML-backend-to-Label-Studio).
