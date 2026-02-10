# EZ-HOI eval-ood 分支

- 未更改 EZ-HOI 结构
- 直接使用原始的 EZ-HOI 解决 OOD 任务

由于作者仅公开了 ViT-L 版本的模型参数，故需要自行训练，具体训练和评测指令如下：

在训练和评测之前，先提取 OOD 数据集的特征：

```shell
python CLIP_ood_extract.py
```

训练指令：

```shell
bash scripts/hico_train_vitB_default.sh
```

评估指令：

```shell
bash scripts/hico_test_vitB_default.sh
```

