
| 时间          | 修改                                                                                                              |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `2025-5-9`  | 创建`v1.0`版本，支持对`onnx`和`tflite`格式的模型量化，当前版本只支持对单个输入的模型进行量化。                                                       |
| `2025-5-10` | `v1.1`版本，输入`onnx`格式模型时，在量化前先对模型进行包括`BN`层融合等在内的优化；`Conv+Activate+BN`，`BN+Activate+Conv`，`GEMM+BN`(这种情况无需融合)无法融合。 |
| `2025-5-11` | `v1.2`版本，修复输入`onnx`模型时，不拆分`BN`层，导致量化结果错误的问题。                                                                    |
| `2025-5-12` | `v2.0`版本，支持多输入模型的量化。                                                                                            |
| `2025-5-14` | `v2.1`版本，解决在将`*.onnx`模型转`*.tflite`模型时，反量化算子融合进`FC`层，导致节点参数变为小数的问题。                                              |
# 1.模型测试结果
![[Performance.png]]
**注**：表中测试的`Weight`和`Activation`都是`8bit`，`B8`指对乘法运算结果先截位，然后再加`8bit Bias`，`B16`指对乘法运算结果先加`16bit Bias`，再截位；针对不同的任务类型，输入缩放系数建议尝试不同的整数值。图像类型任务/卷积神经网络模型的输入缩放系数一般不宜过小，可以在`4`到`32`之间尝试；文本类型任务的缩放系数则可以尝试较小的缩放系数，例如`1`到`8`。

# 2.工具部署
使用`docker`导入镜像。
```shell
docker load < model_quant_v_2_0.tar
```
生成容器命令示例：
```shell
sudo docker run \
--interactive --tty \
--detach \
--name model_quant \
--hostname berial \
--mac-address 02:42:ac:11:00:02 \
--privileged=true \
--volume /home/berial/docker/temp/home:/temp \
-p 5901:5901 \
model_quant:v2.0
```
注：`--volume`后`:`前为宿主机上的文件夹路径，`:`后为容器内文件夹的路径，作用与虚拟机里的共享文件夹相同。
工具在路径`/workdir/model_quantify`下。

# 3.工程组织结构
```shell
.
|-- Quany_V1.so
|-- main.py
|-- model
|   |-- MBV1_NLB_BN_ONLY.onnx
|   |-- MBV1_NLB_BN_ONLY.tflite
|   |-- MMoE_best_model.onnx
|   `-- MMoE_best_model.tflite
|-- model_quantify.so
`-- tf2tflite.so

1 directory, 8 files
```

# 4.输入参数说明
| 参数              | 说明                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| `--input`       | 指定输入的模型文件路径。                                                                                                    |
| `--output`      | 指定量化模型的输出文件夹。                                                                                                   |
| `--input_scale` | 缩放系数。针对不同的任务类型，输入缩放系数建议尝试不同的整数值。图像类型任务/卷积神经网络模型的输入缩放系数一般不宜过小，可以在`4`到`32`之间尝试；文本类型任务的缩放系数则可以尝试较小的缩放系数，例如`1`到`8`。 |
| `--bias_mode`   | `Bias`量化位数。当前量化的`Weight`和`Activation`都是`8bit`，`B8`指对乘法运算结果先截位，然后再加`8bit Bias`，`B16`指对乘法运算结果先加`16bit Bias`，再截位。  |

# 5.使用示例
示例`1`:
```shell
python3 main.py --input model/MBV1_NLB_BN_ONLY.tflite --output output --input_scale 10 --bias_mode B16
```
输入`*.tflite`文件，执行完成后得到量化后的`*.tflite`模型
```shell
root@berial:/workdir/model_quantify# ls output/
MBV1_NLB_BN_ONLY_quant.tflite
```
示例`2`:
```shell
python3 main.py --input model/MBV1_NLB_BN_ONLY.onnx --output output --input_scale 10 --bias_mode B16
```
输入`*.onnx`文件，执行完成后得到量化后的`*.onnx`模型
```shell
root@berial:/workdir/model_quantify# ls output/
MBV1_NLB_BN_ONLY_quant.onnx  MBV1_NLB_BN_ONLY_quant.tflite
```
