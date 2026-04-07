# HPC friendly running of Holohub
## Convert docker image to apptainer image
on local machine
```shell
apptainer build holohub-surgical-scene-recon.sif holohub-surgical-scene-recon:latest
```
## Cluster tricks

run on login node to download models

```shell
singularity exec --nv --cleanenv  \
--bind $(pwd):/workspace/holohub \
--bind $HF_HOME:/root/.cache/huggingface \
--pwd /workspace/holohub \
holohub.sif \
./holohub run surgical_scene_recon verify_train --local --language python --dryrun --verbose
```

run dummy inference for vggt download
```shell
singularity exec --nv --cleanenv --bind $(pwd):/workspace/holohub --bind $HF_HOME:/root/.cache/huggingface --pwd /workspace/holohub holohub.sif /usr/bin/python applications/surgical_scene_recon/models/vggt/vggt_inference.py --image-dir data/surgical_scene_recon/frames/ --output-dir build/surgical_scene_recon/applications/surgical_scene_recon/output/test
```

[Training] Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /leonardo/home/userexternal/mnunzian/.cache/torch/hub/checkpoints/vgg16-397923af.pth

resolved by scping the file from local machine to cluster


## Training
```shell
singularity exec --nv --cleanenv  \
--bind $(pwd):/workspace/holohub \
--bind $HF_HOME:/root/.cache/huggingface \
--pwd /workspace/holohub \
holohub.sif \
./holohub run surgical_scene_recon full --local --language python
```
--local flag to avoid building docker image, uses current environment instead (inside sif file) 

to work in interactive container
`singularity shell --nv --cleanenv  --bind $(pwd):/workspace/holohub --bind $HF_HOME:/root/.cache/huggingface --pwd /workspace/holohub holohub.sif /bin/bash/ `

then:
`./holohub run surgical_scene_recon verify_train --local `
        --output-dir /workspace/output


### Rendering possible on local machine only
o render on local machine, scp the output files from cluster to local machine and run
`scp -r leonardo:/leonardo_work/IscrC_FLAC/holohub/build/surgical_scene_recon/applications/surgical_scene_recon/output/ ~/Desktop/`

`scp -r ~/Desktop/output matteo@172.28.107.111:/home/matteo/holohub/build/surgical_scene_recon/applications/surgical_scene_recon/`

then run on workstation with gpu:
`./holohub run surgical_scene_recon render --language python`
