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
    train = '/root/datasets/JESTER-V1/annotations/jester-v1-train.txt'
    val = '/root/datasets/JESTER-V1/annotations/jester-v1-validation.txt'
    test = '/root/datasets/JESTER-V1/annotations/jester-v1-test.txt'
    
    new_train = '/root/datasets/JESTER-V1/annotations/jester-v1-train-small.txt'
    new_val = '/root/datasets/JESTER-V1/annotations/jester-v1-validation-small.txt'
    new_test = '/root/datasets/JESTER-V1/annotations/jester-v1-test-small.txt'
    
    train = 'D:\Khanh\Others\Hand_Gesture\datasets\JESTER-V1\\annotations\jester-v1-train.txt'
    val = 'D:\Khanh\Others\Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation.txt'
    test = 'D:\Khanh\Others\Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test.txt'
    
    new_train = 'D:\Khanh\Others\Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-train-small.txt'
    new_val = 'D:\Khanh\Others\Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-validation-small.txt'
    new_test = 'D:\Khanh\Others\Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-test-small.txt'
    
    create_short_version(train, new_train)
    create_short_version(val, new_val)
    create_short_version(test, new_test)
    
if __name__ == '__main__':
    main()