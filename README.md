<h1 align="center">
   🎉🎉  NeurIPS 2025:  Consistent Story Generation: Unlocking the Potential of Zigzag Sampling
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
</div>
<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#application">Application</a> •
  <a href="#visualization">Visualization</a> •
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

## Application

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
  A fiery illustration of A majestic phoenix rising from the ashes in a burst of fire, appearing in a flash of flame, flying through a fiery sky, resting on a burning tree.
  </p>

<p align="center"> 
  <img src="videos/fairy_tales_31_1.gif" width="20%" />
  <img src="videos/fairy_tales_31_2.gif" width="20%" />
  <img src="videos/fairy_tales_31_3.gif" width="20%" />
  <img src="videos/fairy_tales_31_4.gif" width="20%" />
</p>

<p align="center">
  A hyper-realistic digital painting of A cat in a busy alley, on a sled, wearing a bow tie, wearing a small bell.
  </p>
<p align="center"> 
  <img src="videos/animals_17_1.gif" width="20%" />
  <img src="videos/animals_17_2.gif" width="20%" />
  <img src="videos/animals_17_3.gif" width="20%" />
  <img src="videos/animals_17_4.gif" width="20%" />
</p>

<p align="center">
  A watercolor illustration of A male child with a round face, short ginger hair, and curious, wide eyes building a fort, in a backyard,  in a treehouse, playing with a puppy.
</p>
<p align="center"> 
  <img src="videos/humans_17_1.gif" width="20%" />
  <img src="videos/humans_17_2.gif" width="20%" />
  <img src="videos/humans_17_3.gif" width="20%" />
  <img src="videos/humans_17_4.gif" width="20%" />
</p>


<p align="center">
  A hyper-realistic digital painting of A woman with a slender figure, straight red hair, and freckles across the nose at a seaside pier, attending a fair, baking cookies, in a rose garden.
</p>
<p align="center"> 
  <img src="videos/humans_13_1.gif" width="20%" />
  <img src="videos/humans_13_2.gif" width="20%" />
  <img src="videos/humans_13_3.gif" width="20%" />
  <img src="videos/humans_13_4.gif" width="20%" />
</p>



## Citation
If you find our work usefull, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{
li2025consistent,
title={Consistent Story Generation: Unlocking the Potential
of Zigzag Sampling},
author={Mingxiao Li, Mang Ning, Marie-Francine Moens},
booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://arxiv.org/abs/2506.09612}
}
```
