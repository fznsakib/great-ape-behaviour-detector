'''''''''''''''''''''''''''''''''''''''''''''
Imports
'''''''''''''''''''''''''''''''''''''''''''''

torch.backends.cudnn.benchmark = True

# Check if GPU available, and use if so. Otherwise, use CPU
if torch.cuda.is_available(): 
    DEVICE = torch.device("cuda")
else:                         
    DEVICE = torch.device("cpu")
    
'''''''''''''''''''''''''''''''''''''''''''''
Argument Parser
'''''''''''''''''''''''''''''''''''''''''''''

default_dataset_dir = f'{os.getcwd()}/mini_dataset/'

parser = argparse.ArgumentParser(
    description="A spatial & temporal-based two-stream convolutional neural network for recognising great ape behaviour.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)

'''''''''''''''''''''''''''''''''''''''''''''
Main
'''''''''''''''''''''''''''''''''''''''''''''

def main(args):
    
    
    
'''
Things to do:

0. Read and understand Zisserman's paper!
1. Generate optical flow images
2. Create custom dataloader
3. Bring over general structure from ADL coursework
4. Define motion CNN
5. Define spatial CNN

'''