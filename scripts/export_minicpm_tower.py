#!/usr/bin/env python3
"""Export trained MiniCPM-V 4.7 vision+merger without loading its LLM."""
import argparse
from megatoken.omni.perception import export_minicpm_submodules

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-snapshot",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(export_minicpm_submodules(args.parent_snapshot,args.output))

if __name__=="__main__":
    main()
