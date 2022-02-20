# Pymaf reimplementation
> Levon 2021.12.27

## bugs to fix

### 由于 licence 问题导致的 h36m mosh 数据无法获得

```shell
h36m/
    h36m.npy
    h36m_mosh_train.npz
    h36m_mosh_valid_p2.npz
```

```shell
mkdir data
cd data
wget http://visiondata.cis.upenn.edu/volumetric/h36m/h36m_annot.tar
tar -xf h36m_annot.tar
rm h36m_annot.tar

# Download H36M images
mkdir -p h36m/images
cd h36m/images
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S1.tar
tar -xf S1.tar
rm S1.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S5.tar
tar -xf S5.tar
rm S5.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S6.tar
tar -xf S6.tar
rm S6.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S7.tar
tar -xf S7.tar
rm S7.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S8.tar
tar -xf S8.tar
rm S8.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S9.tar
tar -xf S9.tar
rm S9.tar
wget http://visiondata.cis.upenn.edu/volumetric/h36m/S11.tar
tar -xf S11.tar
rm S11.tar
cd ../../..
```

## 异常复杂的数据情况


### train

+ center, scale 共同决定 bounding box,
+ pose, shape 是 SMPL 参数 
+ part 是 2D keypoint, 有 24 个点， 但也是三维的 ; 相应 openpose 也是 2D keypoint 但有 25 个点
+ S 是 3D skeleton， 有 24 个点， 但是四维的

DS | length | imgname (str) | center (2) | scale (1) | pose (72) | shape (10) | part (24, 3) 2D 关节点 | S (24, 4) 3D关节点| openpose | has_smpl | maskname | partname | gender
:---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :----
h36m | 312188 | S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg | 1 | 1 | 1 | 1 | 1 | 1 | | | | | 
mpii | 14810 (unique 9899) | images/015601864.jpg | 1 | 1 |  |  | 1 |  | 1 | | | | 
lsp-orig | 1000 | images/im0001.jpg | 1 | 1 |  |  | 1 |  | 1 | | | |
lspet | 9428 | im00001.png | 1 | 1 |  |  | 1 |  | 1 | | | |  
mpi-inf-3dhp | 96507 | S1/Seq1/imageFrames/video_0/frame_000001.jpg | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | | | 
coco | 28344(unique 18127) | train2014/COCO_train2014_000000044474.jpg | 1 | 1 |  |  | 1 |  | 1 | | | | 

### eval

DS | length | imgname | center | scale | pose | shape | part | S | openpose | has_smpl | maskname | partname | gender
:---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :----
h36m-p1 | 109867 | S9/S9_Directions_1.54138969/S9_Directions_1.54138969_000001.jpg | 1 | 1 |  |  |  | 1 |  |  |  |  | 
h36m-p2 | 27558 | S9/S9_Directions_1.60457274/S9_Directions_1.60457274_000001.jpg | 1 | 1 |  |  |  | 1 |  |  |  |  | 
h36m-p2-mosh | 26859 | S9/S9_Directions_1.60457274/S9_Directions_1.60457274_000001.jpg | 1 | 1 | 1 | 1 | 1 | 1 |  |  |  |  | 
lsp | 1000 | images/im1001.jpg | 1 | 1 |  |  | 1 |  |  |  | 1 | 1 | 
mpi-inf-3dhp | 2929 | TS1/imageSequence/img_000001.jpg | 1 | 1 |  |  | 1 | 1 |  |  |  |  | 
coco | 50197(unique 19010) | val2014/COCO_val2014_000000537548.jpg | 1 | 1 |  |  | 1 |  |  |  |  |  | 
3dpw | 35515(unique 24547) | imageFiles/downtown_enterShop_00/image_00000.jpg | 1 | 1 | 1 | 1 |  |  |  |  |  |  | 1

## Bugs to fix

+ [ ] 当设置 mixture 时出现问题，使得 LSP_ori 等不包含 pose, shape, has_smpl 的数据集，出现了 has_sml datas 中存在问题, 问题出在了 runner.FitsDict
+ OpenDRRander 替换为了 pyRender
+ 用 20000 数据训练之后发现，eval 效果太差，而且人偏瘦。
+ [x] 原版模型测试可以达到论文标称结果，因为之前 test() 里面没有写 self.model.eval(). 