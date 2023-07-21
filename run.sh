export CUDA_VISIBLE_DEVICES=0
# nohup python web_demo_moss.py --model_dir /data/moss/moss-moon-003-sft --port 8087 2>&1 &
nohup python web_demo_llama.py --model_dir ../Chinese-Alpaca-Plus-7B --port 8087 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python web_demo_chatglm.py --model_dir ../chatglm-6b --port 8088 2>&1 &
export CUDA_VISIBLE_DEVICES=2
nohup python web_demo_chatglm.py --model_dir ../chatglm2-6b --port 8089 2>&1 &
nohup python gpt35.py --port 8086 2>&1 &
