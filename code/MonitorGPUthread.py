import GPUtil
import time
import os

commands = []

command1 = 'python main.py --model frvd --n_frames 7 --model_label 1--loss_denoise "1.0*MSE" --save_gt --save_results   --save "frvd-v1.0"  --data_range "1-800/801-824" --epoch 100'
commands.append(command1)
command2 = 'python main.py --model frvdwof --n_frames 7 --model_label 2  --loss_denoise "1.0*MSE" --loss_flow "1.0*L1+1.25*TVL1" --save_result --save_gt --save_of --save "frvdwof-v0.3" --data_range "1-800/801-824" --epoch 100'
commands.append(command2)
command3 = 'python main.py --model frvdwof --n_frames 7 --model_label 2  --loss_denoise "1.0*MSE" --loss_flow "2.0*L1+2.5*TVL1" --save_result --save_gt --save_of --save "frvdwof-v0.4" --data_range "1-800/801-824" --epoch 100'
commands.append(command3)
command4 = 'python main.py --model frvdwof --n_frames 7 --model_label 2  --loss_denoise "1.0*MSE" --loss_flow "3.0*L1+3.75*TVL1" --save_result --save_gt --save_of --save "frvdwof-v0.5" --data_range "1-800/801-824" --epoch 100'
commands.append(command4)
command5 = 'python main.py --model frvdwof --n_frames 7 --model_label 2  --loss_denoise "1.0*MSE" --loss_flow "3.0*L1+6.0*TVL1" --save_result --save_gt --save_of --save "frvdwof-v0.6" --data_range "1-800/801-824" --epoch 100'
commands.append(command5)

command_idx = 0

while(True):
    try:
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        print(command[command_idx])
        os.system(command)
        command_idx += 1
        if command_idx >= len(commands):
            break
    except:
        print ('=================GPU Information====================')
        print ("Executing Command", command_idx)
        print ("Waiting GPU Free...")
        print (time.strftime("%F") + ' ' +  time.strftime("%T"))
        print ('====================================================')
        time.sleep(1 * 60 * 30)


print('Done!!')
