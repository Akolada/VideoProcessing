# Recycle-GAN

## Summary

![Summary](https://github.com/SerialLain3170/VideoProcessing/blob/master/Recycle-GAN/summary.png)
- CycleGANの拡張版
- Recurrent lossとRecycle lossを考慮、PredictorにはUNetを用いている。

## Usage
`x_path`と`y_path`にそれぞれ抽出したフレーム画像を入れて
```
$ python train.py
```
を実行する

## Result
私の環境で生成したgif動画を以下に示す。

![Result](https://github.com/SerialLain3170/VideoProcessing/blob/master/Recycle-GAN/mtou.gif)

- Predictorの入力には現時刻とその前の時刻のフレームをconcatして用いた
- Recycle lossとRecurrent loss、Cycle lossの重みは10.0
- 最適化手法はAdam(α=0.0002、β
