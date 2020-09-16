export CUDA_VISIBLE_DEVICES=7
export FLAGS_fraction_of_gpu_memory_to_use=0.2

algo=KL                           # set KL, abs_max or min_max
is_full_quantize=true            # str, set True or False 
weight_bits=8
activation_bits=8
activation_quantize_type="moving_average_abs_max" # moving_average_abs_max range_abs_max
weight_quantize_type="abs_max"
quantizable_op_type="conv2d,depthwise_conv2d,mul"
use_gpu=true
optimize_model=false
use_slim=false
batch_size=2
batch_nums=100

python slim/quantization/post_quantize.py \
    --config_path="./configs/yolov3_mobilenet_v1.yml" \
    --model_name="mobilenetv1_yolov3" \
    --model_path="./slim/quantization/yolov3_mobilenet_v1_coco_608_fp32_fluid_opt" \
    --algo=${algo} \
    --output_path="./slim/quantization/mobilenetv1_yolov3_quant" \
    --is_full_quantize=${is_full_quantize} \
    --weight_bits=${weight_bits} \
    --activation_bits=${activation_bits} \
    --activation_quantize_type=${activation_quantize_type} \
    --weight_quantize_type=${weight_quantize_type} \
    --quantizable_op_type=${quantizable_op_type} \
    --use_gpu=${use_gpu} \
    --batch_size=${batch_size} \
    --batch_nums=${batch_nums} \
    --optimize_model=${optimize_model} \
    --use_slim=${use_slim}
