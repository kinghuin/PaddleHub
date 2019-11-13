python ../../paddlehub/commands/hub.py autofinetune text_classifier.py --param_file=autodl.yaml --gpu=0,1,2,3,4,5,6,7 --popsize=16 --round=10 \
 --output_dir=./autodl.output.lcqmc.1113 --evaluator=fulltrail --tuning_strategy=hazero dataset lcqmc
