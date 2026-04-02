# HPC friendly running of Holohub
## convert docker image to apptainer image
on local machine
```shell
apptainer build holohub-surgical-scene-recon.sif holohub-surgical-scene-recon:latest
```
## run on cluster

Modify path to holohub and hf_cache as needed

```shell
singularity exec --nv --cleanenv
--bind $(pwd):/workspace/holohub
--bind $(HF_HOME):/root/.cache/huggingface
--pwd /workspace/holohub
holohub.sif
./holohub run surgical_scene_recon verify_train --local --language python --dryrun --verbose
```
