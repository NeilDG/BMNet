#Script to use for running heavy training.

import os

def main():
    # os.system("python ColorTrans/train_custom.py --opt ./ColorTrans/options/train/train_Enhance.yml")
    # os.system("python MainNet/train_custom.py --opt MainNet/options/train/train_Enhance_ISTD.yml")
    # os.system("shutdown /s /t 1")

    os.system("python test_custom.py --opt MainNet/options/train/test_Enhance_ISTD.yml")

if __name__ == "__main__":
    main()
