import argparse
from alzheimer_agent.projection import project_checkpoint

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    print(project_checkpoint(a.checkpoint,a.manifest,a.output,device=a.device))
