import os

avi_dir = 'video'
out_dir = 'frame'

for avi in os.listdir(avi_dir):
    print('extract the frame of ' + avi)
    avi_path = os.path.join(avi_dir, avi)
    out_path = os.path.join(out_dir, avi[:-4])

    os.makedirs(out_path, exist_ok=True)
    command_extract = "select=not(mod(n\,%d))" % (30)  #每隔几帧输出一帧
    mes = r'ffmpeg -i ' + avi_path + ' -vf "%s" -vsync 0 ' % command_extract + \
        out_path + '/'  + '%d.jpg'
    # print(mes)
    os.system(mes)