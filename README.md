<h1 align="center">
   🎉  NeurIPS 2025 🎉  Consistent Story Generation: Unlocking the Potential of Zigzag Sampling
  <br>
</h1>

<div align="center">

<!-- <a href="http://59.110.93.62:7089/" style="display: inline-block;">
    <img src="./resource/gradio.svg" alt="demo" style="height: 20px; vertical-align: middle;">
</a>&nbsp; -->
<a href="https://arxiv.org/pdf/2506.09612.pdf" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;
<!-- <a href="https://byliutao.github.io/1Prompt1Story.github.io/" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Project_page-More_visualizations-green" alt="project page" style="height: 20px; vertical-align: middle;">
</a>&nbsp; -->
<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#more application">Application</a> •
  <a href="#visualizations">Visualization</a> 
  <a href="#citation">Citation</a> •
</p>


## How To Use
```bash
# Clone this repository
$ git clone https://github.com/Mingxiao-Li/Asymmetry-Zigzag-StoryDiffusion.git

# Go into the repository
$ cd Asymmetry-Zigzag-StoryDiffusion

### Install dependencies ###
$ conda env create -f environment.yaml
$ conda activate zigzagstory

# Run infer code
$ sh run_flux_img.sh

```

## Applications 

* Visual story generation with consistent subjects across multiple images.
 ```bash 
  story_image_generation.py
 ```
* In combination with the TI2V model (Text-and-Image-to-Video), enables the creation of video stories featuring consistent subjects throughout.
 ```bash 
  story_video_generation.py
 ```

## Visualization
### Story (Flux)
![flowchar-img](images/flux_more_vis.png)

### Story (SDXL)
![flowchar-img](images/sdxl-more-vis.png)

### Comparison (Flux)
![flowchar-img](images/flux_compar1.png)

### Comparison (SDXL)
![flowchar-img](images/sdxl_compar.png)


### Long Story (FLUX)
![flowchar-img](images/long_story_1.png)

### Video Story (FLUX + Wan2.1 TI2V)

<p align="center">

  <figure style="display:inline-block; text-align:center; margin:10px;">
    <figcaption>rising from the ashes in a burst of fire</figcaption>
    <video src="./videos/fairy_tales_31_1mp4.mp4" width="22%" controls></video>
  </figure>

  <figure style="display:inline-block; text-align:center; margin:10px;">
    <figcaption>appearing in a flash of flame</figcaption>
    <video src="video2.mp4" width="22%" controls></video>
  </figure>

</p>


## Citation
If you find our work usefull, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{
liu2025onepromptonestory,
title={One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt},
author={Tao Liu and Kai Wang and Senmao Li and Joost van de Weijer and Fhad Khan and Shiqi Yang and Yaxing Wang and Jian Yang and Mingming Cheng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=cD1kl2QKv1}
}
```
