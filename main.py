import Quany_V1
import tf2tflite
import model_quantify
import argparse

# e.g.
# python3 main.py --input model/MMoE_best_model.tflite --output output --input_scale 10 --bias_mode B16
# python3 main.py --input model/MMoE_best_model.tflite --output output --input_scale 20 --bias_mode B16
# python3 main.py --input model/vgg_16_bn.onnx --output output --input_scale 16 --bias_mode B16
# python3 main.py --input model/densenet_40.onnx --output output --input_scale 16 --bias_mode B16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Quantization Tool (ONNX/TFLite)")
    parser.add_argument("--input", type=str, required=True, help="Path to input model (ONNX/TFLite)")
    parser.add_argument("--output", type=str, required=True, help="Path to save quantized (ONNX/TFLite) model")
    parser.add_argument("--input_scale", type=int, default=16, help="Input scaling factor (any positive integer)")
    parser.add_argument("--bias_mode", type=str, default="B16", choices=["B8", "B16"],
                        help="Bias quantization mode")
    parser.add_argument("--weight_bits", type=int, default=8, choices=[8],
                        help="Number of bits for weight quantization")
    parser.add_argument("--act_bits", type=int, default=8, choices=[8],
                        help="Number of bits for activation quantization")

    args = parser.parse_args()

    model_quantify.run(args)
