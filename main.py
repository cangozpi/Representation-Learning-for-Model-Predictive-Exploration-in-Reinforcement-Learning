from config import *

if __name__ == '__main__':
    if args['train']:
        from train import main
    elif args['eval']:
        from eval import main
    main(args)