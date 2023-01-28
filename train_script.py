#Script to use for running heavy training.

import os

def main():
    os.system("python ColorTrans/train_custom.py --opt ColorTrans/options/train/train_Enhance.yml")

    os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
