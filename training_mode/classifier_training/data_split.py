
def split(classes_file, img_list, train_target_file, test_target_file, train_count= 10):
    names_list = {}
    names_list_idxs = []
    classes_file_buf = open(classes_file)
    line = classes_file_buf.readline().strip()
    while line:
        if line not in names_list:
            names_list[line] = 0
            names_list_idxs.append(line)
        line = classes_file_buf.readline().strip()
    
    img_list_buf = open(img_list)
    line = img_list_buf.readline().strip()
    train_list = []
    test_list = []
    train_file = open(train_target_file, 'w')
    test_file = open(test_target_file, 'w')
    while line:
        image_path = line.split(' ')[0]
        image_name = image_path.split('/')[0]
        if image_name in names_list:
            image_label = names_list_idxs.index(image_name)
            if names_list[image_name] < train_count:
                train_list.append((image_path, int(image_label)))
                names_list[image_name] += 1
                train_file.write(line + '\n')
            else:
                test_list.append((image_path, int(image_label)))
                test_file.write(line + '\n')
        line = img_list_buf.readline().strip()
    
if __name__ == "__main__":
    train_file = '../../data/test/lfw/127_names_img_list_train.txt' 
    test_file = '../../data/test/lfw/127_names_img_list_test.txt' 
    classes_file = '../../data/test/lfw/127_names.txt' 
    img_list = '../../data/test/lfw/img_list.txt' 
    split(classes_file, img_list, train_file, test_file)