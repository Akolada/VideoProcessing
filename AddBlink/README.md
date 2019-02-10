# Adding Blink to single image with using optical flow
リポジトリ名FlowTextureGANと書いてありますが、単にOptical flowを用いて瞬き付与を行っただけです。

## Summary
![here](https://github.com/SerialLain3170/VideoProcessing/blob/master/AddBlink/network.png)

- Reference VideoからOptical Flow Sequenceを作り、Conditionとして与えます。2D UNetと3D UNetを通して動画にします
- 顔画像全体を使うのではなく、目だけを抽出して目だけの動画を生成しています。

## Results
私の環境で生成した動画は以下になります。右が生成動画です。
![here](https://github.com/SerialLain3170/VideoProcessing/blob/master/AddBlink/cv_generation.gif)

- バッチサイズは2
- 最適化手法はAdam(α=0.0002, β1=0.5)
