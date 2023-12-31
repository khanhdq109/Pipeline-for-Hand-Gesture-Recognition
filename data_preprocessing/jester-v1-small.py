import sys
import random

def create_short_version(file, new_file, ratio = 0.01):
    # Open file
    with open(file, 'r') as f:
        # Read all lines
        lines = f.readlines()
    # Randomly select lines
    selected_lines = random.sample(lines, int(len(lines) * ratio))
    # Create a new file and write the selected lines to it
    with open(new_file, 'w') as f:
        f.writelines(selected_lines)
    print('Successfully!!!')
    print('Number of samples before: {}'.format(len(lines)))
    print('Number of samples after: {}'.format(len(selected_lines)))
        
def main():
    print('Create a small version of JESTER-V1 dataset')
    # Get ratio
    arg = sys.argv[1]
    arg = str.upper(arg)
    ratio = float(arg)
    
    train = '../../datasets/JESTER-V1/annotations/jester-v1-train.txt'
    val = '../../datasets/JESTER-V1/annotations/jester-v1-validation.txt'
    test = '../../datasets/JESTER-V1/annotations/jester-v1-test.txt'
    
    new_train = '../../datasets/JESTER-V1/annotations/jester-v1-train-small.txt'
    new_val = '../../datasets/JESTER-V1/annotations/jester-v1-validation-small.txt'
    new_test = '../../datasets/JESTER-V1/annotations/jester-v1-test-small.txt'
    
    create_short_version(train, new_train, ratio = ratio)
    create_short_version(val, new_val, ratio = ratio)
    create_short_version(test, new_test, ratio = ratio)
    
    print('Create a small version of JESTER-V1 dataset successfully!!!')
    
if __name__ == '__main__':
    main()