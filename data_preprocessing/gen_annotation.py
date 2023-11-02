import os

def gen_label(
    src = 'csv/jester-v1-labels.csv',
    # des = '/root/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-labels.txt'
    des = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations/jester-v1-labels.txt' # Delete
):
    print('Generating label file...')
    with open(src, 'r') as fcsv:
        with open(des, 'w') as ftxt:
            for line in fcsv:
                line = line.strip()
                ftxt.write(line + '\n')
    print('Generate label file successfully!!!')
    
def gen_annotation(
    # path = '/root/Hand_Gesture/datasets/JESTER-V1/annotations',
    path = 'D:/Khanh/Others/Hand_Gesture/datasets/JESTER-V1/annotations', # Delete
    mode = 'train'
):
    # Choose mode
    if mode == 'train':
        print('Generating train file...')
        src = 'csv/jester-v1-train.csv'
        des = os.path.join(path, 'jester-v1-train.txt')
    elif mode == 'val':
        print('Generating val file...')
        src = 'csv/jester-v1-validation.csv'
        des = os.path.join(path, 'jester-v1-validation.txt')
    elif mode == 'test':
        print('Generating test file...')
        src = 'csv/jester-v1-test.csv'
        des = os.path.join(path, 'jester-v1-test.txt')
    else:
        raise ValueError('Invalid mode!')
    # Get list of label
    label_list = []
    with open(os.path.join(path, 'jester-v1-labels.txt'), 'r') as f:
        for line in f:
            label_list.append(line.strip())
    # Generate annotation file
    with open(src, 'r') as fcsv:
        with open(des, 'w') as ftxt:
            for line in fcsv:
                line = line.strip()
                if mode == 'train' or mode == 'val':
                    line = line.split(';')
                    ftxt.write(line[0] + ' ' + str(label_list.index(line[1])) + '\n')
                elif mode == 'test':
                    ftxt.write(line + '\n')
    print('Generate ' + mode + ' successfully!!!')
    
def main():
    gen_label()
    gen_annotation(mode = 'train')
    gen_annotation(mode = 'val')
    gen_annotation(mode = 'test')
    
if __name__ == '__main__':
    print('\nStarting gen_annotation.py')
    
    main()
    
    print('Generate annotation completely')
            