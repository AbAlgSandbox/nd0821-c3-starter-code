import os
import subprocess
import argparse


def print_slicing_cycle(feature):

    slicing_result = subprocess.run(['python', 'slicing_cycle.py', '--feature', feature], capture_output=True, text=True)

    slicing_output = slicing_result.stdout
    
    errors_produced = slicing_result.stderr
    
    print(errors_produced)

    with open(os.path.join(os.getcwd(), 'slice_output.txt'), 'w') as file:
        file.write(slicing_output)
    
    return


def go(args):
    
    feature = args.feature
    
    print_slicing_cycle(feature)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cycle feature to obtain performance statistics")


    parser.add_argument(
        "feature",
        help="Feature through which unique values to cycle"
    )


    args = parser.parse_args()

    go(args)