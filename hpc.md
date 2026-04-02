# HPC friendly running of Holohub
## convert docker image to apptainer image
```shell
apptainer build holohub-surgical-scene-recon.sif holohub-surgical-scene-recon:latest
```
## run on cluster

Modify path to holohub and hf_cache as needed

```shell
apptainer exec --nv --cleanenv
--bind /path/to/holohub:/workspace/holohub
--bind /scratch/$USER/hf_cache:/root/.cache/huggingface
--pwd /workspace/holohub
holohub-surgical-scene-recon.sif
./holohub run surgical_scene_recon verify_train --local --language python --dryrun --verbose
```
