import random

def create_short_version(file, new_file, ratio = 0.05):
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
    ratio = 0.2
    
    # train = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-train.txt'
    # val = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation.txt'
    # test = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test.txt'
    
    # new_train = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-train-small.txt'
    # new_val = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation-small.txt'
    # new_test = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test-small.txt'
    
    train = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-train.txt' # Delete
    val = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation.txt' # Delete
    test = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test.txt' # Delete
    
    new_train = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-train-small.txt' # Delete
    new_val = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation-small.txt' # Delete
    new_test = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test-small.txt' # Delete
    
    create_short_version(train, new_train, ratio = ratio)
    create_short_version(val, new_val, ratio = ratio)
    create_short_version(test, new_test, ratio = ratio)
    
if __name__ == '__main__':
    main()