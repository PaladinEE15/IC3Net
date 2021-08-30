#!/bin/bash

python main.py --env_name traffic_junction --nagents 20 --nprocesses 8 --num_epochs 100 --hid_size 128 --detach_gap 10 --lrate 0.0003 --dim 18 --max_steps 60 --ic3net --vision 0 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 20 --curr_end 60 --difficulty hard --load 'models/tj_hard/rawB/model_20' --test_times 1000 --quant_levels 30 --test_quant

python main.py --env_name traffic_junction --nagents 20 --nprocesses 8 --num_epochs 100 --hid_size 128 --detach_gap 10 --lrate 0.0003 --dim 18 --max_steps 60 --ic3net --vision 0 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 20 --curr_end 60 --difficulty hard --load 'models/tj_hard/rawB/model_20' --test_times 1000 --quant_levels 25 --test_quant

python main.py --env_name traffic_junction --nagents 20 --nprocesses 8 --num_epochs 100 --hid_size 128 --detach_gap 10 --lrate 0.0003 --dim 18 --max_steps 60 --ic3net --vision 0 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 20 --curr_end 60 --difficulty hard --load 'models/tj_hard/rawB/model_20' --test_times 1000 --quant_levels 20 --test_quant

python main.py --env_name traffic_junction --nagents 20 --nprocesses 8 --num_epochs 100 --hid_size 128 --detach_gap 10 --lrate 0.0003 --dim 18 --max_steps 60 --ic3net --vision 0 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 20 --curr_end 60 --difficulty hard --load 'models/tj_hard/rawB/model_20' --test_times 1000 --quant_levels 15 --test_quant

python main.py --env_name traffic_junction --nagents 20 --nprocesses 8 --num_epochs 100 --hid_size 128 --detach_gap 10 --lrate 0.0003 --dim 18 --max_steps 60 --ic3net --vision 0 --recurrent --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 20 --curr_end 60 --difficulty hard --load 'models/tj_hard/rawB/model_20' --test_times 1000 --quant_levels 10 --test_quant