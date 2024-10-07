import zlib
import pickle
import base64
import os

print(os.environ["LD_LIBRARY_PATH"])
ld_library_paths = os.environ["LD_LIBRARY_PATH"].split(':')
mujoco_path = '/home/miko-debian/.mujoco/mujoco210/bin'
if mujoco_path not in ld_library_paths:
    ld_library_paths.append(mujoco_path)

os.environ["LD_LIBRARY_PATH"]=':'.join(ld_library_paths)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('encoded_thunk')
    args = parser.parse_args()
    # print(f'{args.encoded_thunk=}')
    decoded = base64.b64decode(args.encoded_thunk)
    # print(f'{decoded=}')
    # print('')
    # print(f'{zlib.decompress(decoded)=}')
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()