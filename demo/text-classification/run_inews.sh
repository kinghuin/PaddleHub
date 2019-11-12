export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.97

python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=2,3,4,5,6,7 --popsize=12 --round=10 \
 --output_dir=./autodl.output.inews.1112 --evaluator=fulltrail --tuning_strategy=hazero dataset inews
