# HuBERT-VUV

The HuBERT-based model to discriminate voiced / unvoiced frames (20ms/frame).
VUVmodel.predict() returns a digit for one frame: 

0: silence
1: voiced
2: unvoiced

To check out the repository, make sure to pull large files (models) using git lfs:

```{sh}
git clone https://github.com/akinori-ito/HuBERT-VUV
cd HuBERT-VUV
git lfs pull
```
