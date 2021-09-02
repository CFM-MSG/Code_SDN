
# Introduction

This is the implementation code and instruction of the proposed work  *"SDN: Semantic Decoupling Networks for Temporal Language Grounding"* (SDN).

## Semantic Decoupling Networks

Temporal Language Grounding is one of the challenging cross-modal video understanding tasks, which aims at retrieving the most relevant video segment from an untrimmed video according to a natural language sentence.
The existing method can be separated into two dominant types: proposal-free and proposal-basedmethods.
Both two paradigms have respective characters. The former shows advantages on segment-levelinteraction while the latter has a strong ability on localizing timestamps flexibly.

We propose a novel framework termed Semantic Decoupling Networks (SDN), which introduces the benefits of proposal-based methods into the proposal-free framework by exploring the coarse-to-fine semantics in videos. We also propose the Semantic Modeling Block (SMB) to capture thesemantics in decoupled video features and fuse the two modalities.

![avatar](fig/framework.png)

<!-- **Insight of Our Work** -->
## Insight of Our Work

1. We propose the Semantic Decoupling Module and the Semantic Modeling Block for temporal language grounding to explore the semantics in different video segments with multi-scales.  
2. Based on the blocks we proposed, we build a novel proposal-free framework Semantic Decoupling Networks for TLG task that combines the benefits of both proposal-based and proposal-free methods by introducing the semantics in multi-scales video candidates. Since we just model the semantic features in different scales instead of generating many candidates, our SDN still maintains proposal-free methods' efficiency.
3. We conduct extensive experiments on three public datasets in a comparable setting. The experimental results show that our SDN outperforms recent state-of-the-art approaches and demonstrate the rationality of motivation of the semantic modeling.
  
# Training and Testing

## Running

Use the following command to train our model.

```Python
CUDA_VISIBLE_DEVICES=0 python main.py -c exps/TLG/ExpSDN_Charades_VGG.yaml
```

You can resume a checkpoint file by

```Python
CUDA_VISIBLE_DEVICES=0 python main.py -c exps/TLG/ExpSDN_Charades_VGG.yaml --resume $path to *.ckpt$
```

If you just want to evaluate a saved model file, use the following command.

```Python
CUDA_VISIBLE_DEVICES=0 python main.py -c exps/TLG/ExpSDN_Charades_VGG.yaml --test --load $path to *.best$
```

Use the following command to activate multi-gpu training:

```Python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py -c exps/TLG/ExpSDN_Charades_VGG.yaml
```

Remember to change the dataset used at line 44 in main.py if you want to train the model on other datasets.

```Python
-main.py
     -line 44:   dsets, loader = charades_sta_vgg.create_loaders(self.cfg.loader_config)

```

# Overall Results

<!-- **Results on Charades-STA Dataset** -->
## Results on Charades-STA Dataset

![avatar](fig/charades.png)

<!-- **Results on TACoS Dataset** -->
## Results on TACoS Dataset

![avatar](fig/tacos.png)

<!-- **Results on ActivityNet-Caption Dataset** -->
## Results on ActivityNet-Caption Dataset

![avatar](fig/activitynet.png)

<!-- **Visualization of What Our Model Care** -->

## Visualization of Moment-Text Retrieval

![avatar](fig/visualization.png)

# Acknowledge

We sincerely thank the following works for their video features and codes.

```ref
@inproceedings{rodriguez_WACV_2021,
  author    = {Cristian Rodriguez Opazo and
               Edison Marrese{-}Taylor and
               Basura Fernando and
               Hongdong Li and
               Stephen Gould},
  title     = {DORi: Discovering Object Relationships for Moment Localization of
               a Natural Language Query in a Video},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision},
  pages     = {1078--1087},
  year      = {2021},
}

@inproceedings{mun_CVPR_2020,
  author    = {Jonghwan Mun and
               Minsu Cho and
               Bohyung Han},
  title     = {Local-Global Video-Text Interactions for Temporal Grounding},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {10807--10816},
  year      = {2020},
}
​```
