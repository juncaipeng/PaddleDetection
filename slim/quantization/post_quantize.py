import sys
import os
import argparse
import functools
import time
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from utility import add_arguments, print_arguments

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

# inputs
parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("config_path", str, "", "")
add_arg("model_name", str, "", "")
add_arg("model_path", str, "", "")
add_arg("algo", str, "", "")
add_arg("batch_size", int, 10, "")
add_arg("batch_nums", int, 10, "")
add_arg("output_path", str, "", "")
add_arg("is_full_quantize", bool, False, "")
add_arg("is_use_cache_file", bool, False, "")
add_arg("weight_bits", int, 8, "")
add_arg("activation_bits", int, 8, "")
add_arg("activation_quantize_type", str, "", "")
add_arg("weight_quantize_type", str, "", "")
add_arg("quantizable_op_type", str, "", "")
add_arg("use_gpu", bool, False, "")
add_arg("optimize_model", bool, False, "")
add_arg("use_slim", bool, False, "")
args = parser.parse_args()
print_arguments(args)

# prepare
quantizable_op_type = []
for op_type in args.quantizable_op_type.split(","):
    quantizable_op_type.append(op_type)
print("quantizable_op_type:" + str(quantizable_op_type))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
save_model_path = os.path.join(args.output_path,
        args.model_name + "_" + args.algo + "_quant")
print("save_model_path:" + str(save_model_path))

# quantization 
place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

def sample_generator():
    cfg = load_config(args.config_path)
    eval_reader = create_reader(cfg.EvalReader)
    for data_list in eval_reader():
        for data in data_list:
            yield data[0], data[1]

'''
for data in eval_reader():
    print(len(data))
    print(len(data[0]))
    print(data[0])
    exit()
'''
#sample_generator = reader.val(data_dir=args.data_dir)

start = time.time()
if args.use_slim:
    from paddleslim.quant import quant_post
    quant_post(
                executor=exe,
                model_dir=args.model_path,
                quantize_model_path=save_model_path,
                sample_generator=sample_generator,
                model_filename='model',
                params_filename='params',
                save_model_filename=None,
                save_params_filename=None,
                batch_size=args.batch_size,
                batch_nums=args.batch_nums,
                scope=None,
                algo=args.algo,
                quantizable_op_type=quantizable_op_type,
                is_full_quantize=args.is_full_quantize,
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
                activation_quantize_type=args.activation_quantize_type,
                weight_quantize_type=args.weight_quantize_type,
                is_use_cache_file=args.is_use_cache_file
                )
else:
    ptq = PostTrainingQuantization(
                executor=exe,
                sample_generator=sample_generator,
                model_dir=args.model_path,
                model_filename=None,
                params_filename=None,
                batch_size=args.batch_size,
                batch_nums=args.batch_nums,
                algo=args.algo,
                quantizable_op_type=quantizable_op_type,
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
                activation_quantize_type=args.activation_quantize_type,
                weight_quantize_type=args.weight_quantize_type,
                is_full_quantize=args.is_full_quantize,
                optimize_model=args.optimize_model,
                is_use_cache_file=args.is_use_cache_file)
    quantized_program = ptq.quantize()

    ptq.save_quantized_model(save_model_path)
times = time.time() - start

print("post training quantization finish, and it takes " + str(times) + ". \n\n")




