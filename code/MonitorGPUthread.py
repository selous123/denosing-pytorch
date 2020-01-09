import GPUtil
import time
import os
command = 'python main.py --model frvdwof --n_frames 7 --model_label 2  --loss_denoise "4.0*L1" --loss_flow "1.0*L1+1.25*TVL1" --save_result --save_gt --save_of --save "frvdwof-v0.1" --data_range "1-800/801-824"'

while(True):
    try:
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        print(command)
        os.system(command)
        break
    except:
        print ("Waiting GPU Free...")
        print (time.strftime("%F") + ' ' +  time.strftime("%T"))
        time.sleep(1 * 60 * 30)


print('Done!!')
