python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=2,3,4,5,6,7 --popsize=12 --round=10 \
 --output_dir=./autodl.output.xnli.1112 --evaluator=fulltrail --tuning_strategy=hazero dataset xnli
