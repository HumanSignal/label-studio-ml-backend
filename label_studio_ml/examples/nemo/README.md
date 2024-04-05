<!--
---
title: Automatic Speech Recognition with NVIDIA NeMo
type: blog
tier: all
order: 45
meta_title: Automatic Speech Recognition with Label Studio and NVIDIA NeMo
meta_description: This is a tutorial on how to use the example NVIDIA NeMO model backend with Label Studio.
categories:
    - tutorial
    - automatic speech recognition
    - nemo
    - nvidia
image: "/tutorials/nvidia.png"
---
-->

## Quickstart

Build and start the Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up -d
```

Check if it works:

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

Then connect tje running backend to Label Studio:

```bash
label-studio start --init new_project --ml-backends http://localhost:9090 
```

# Reference to tutorial

See the tutorial in the documentation for building your own image and advanced usage:

https://github.com/heartexlabs/label-studio/blob/master/docs/source/tutorials/nemo_asr.md

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [HumanSignal](https://www.humansignal.com/) 2024

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
