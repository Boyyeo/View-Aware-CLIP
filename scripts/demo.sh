

PROMPT='a photo of a fox right side view'
SEED=100

#PROMPT='a photo of a dog left side view'

python inference_scripts/demo.py  --seed $SEED --prompt "${PROMPT}" --method 'stable_only' 
python inference_scripts/demo.py  --seed $SEED --prompt "${PROMPT}" --method 'pretrained_clip'
python inference_scripts/demo.py  --seed $SEED --prompt "${PROMPT}" --method 'view_aware_clip' --clip_ckpt 'saved_checkpoint/view_aware_ckpt.pyt'
python inference_scripts/demo_plot.py --seed $SEED --prompt "${PROMPT}"