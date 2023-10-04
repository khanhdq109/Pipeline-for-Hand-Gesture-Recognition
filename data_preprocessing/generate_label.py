import os
import os.path as osp 

from tqdm import tqdm 

if __name__ == '__main__':
    meta_dataset = 'datas/jester/jester-v1' # metadata of the Jester dataset 
    dataset = '/media/milky/Samsung_T5/20BN-Jester/jester-v1-tmp' # dataset path 

    # == class list loading == # 
    # ---------------------------
    with open(f'{meta_dataset}-labels.csv') as f:
        lines = f.readlines()

    categories = []
    for line in lines:
        line = line.rstrip() # 오른쪽 공백문자 제거 
        categories.append(line)

    categories = sorted(categories) # 제스처 클래스 소팅 

    # == generate 'class list' in .txt  == # 
    # -------------------------------------
    with open('datas/jester/category.txt', 'w') as f:
        f.write('\n'.join(categories))

    # == class_name to index == # 
    # ---------------------------
    """ {'class_name': index_num}
    """
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    
    # == generate train and validate dataset list in .txt == # 
    # --------------------------------------------------------
    files_input = [f'{meta_dataset}-validation.csv', f'{meta_dataset}-train.csv']  # 데이터 폴더 리스트 
    files_output = ['datas/jester/val_videofolder.txt', 'datas/jester/train_videofolder.txt'] # 저장 파일 이름 
    
    for (filename_input, filename_output) in zip(files_input, files_output):

        with open(filename_input) as f:
            lines = f.readlines()
        
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';') # [디렉토리_번호, 제스처 레이블]
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]]) # 클래스 인덱스 


        output = []
        for i in tqdm(range(len(folders))):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            dir_files = os.listdir(osp.join(dataset, curFolder))
            output.append('%s %d %d' % (osp.join(dataset, curFolder), len(dir_files), curIDX))
            
        with open(filename_output, 'w') as f:
            """ 데이터경로, 이미지 개수, 레이블 인덱스 
            """
            f.write('\n'.join(output))

    # == test dataset == # 
    # ----------------------
    with open(f'{meta_dataset}-test.csv') as f:
        lines = f.readlines()

    folders = []
    for line in lines:
        folders.append(line.strip())

    output = []
    for i in tqdm(range(len(folders))):
        curFolder = folders[i]
        dir_files = os.listdir(osp.join(dataset, curFolder))
        output.append('%s %d' % (osp.join(dataset, curFolder), len(dir_files)))
        
    with open('datas/jester/test_videofolder.txt', 'w') as f:
        f.write('\n'.join(output))
