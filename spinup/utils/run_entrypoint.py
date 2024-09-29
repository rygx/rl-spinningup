import zlib
import pickle
import base64

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