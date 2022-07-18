import os
import numpy as np

body_hands = [2, 4, 11, 21, 58, 77, 82, 98, 107, 128, 150, 151, 257, 383, 289, 290, 305, 334, 335, 344]

athletics_jumping = [152, 161, 183, 208, 254, 368]
some_dance = [35, 85, 278, 343]

hand = [82, 98, 383]
head_mouse = [13, 17, 27, 38, 317, 320, 393]
eat_drink = [92, 101, 111, 115, 204, 354]
makeup = [5, 109, 127, 10]
arts_crafts = [25, 36, 99, 391]
athletics_throw_launch = [6, 299]
cloth = [97, 132, 187, 371]
heights = [1, 68, 173, 344]

extract_label = []
extract_label.append(body_hands)
extract_label.append(athletics_jumping)
extract_label.append(some_dance)
extract_label.append(hand)
extract_label.append(head_mouse)
extract_label.append(eat_drink)
extract_label.append(makeup)
extract_label.append(arts_crafts)
extract_label.append(athletics_throw_launch)
extract_label.append(cloth)
extract_label.append(heights)
extract_label = np.concatenate(extract_label)
extract_label = np.unique(extract_label)

# x = [i + 1 for i in range(400)]
# rest_set = set(x) - set(extract_label)
# rest_set = np.array(list(rest_set))
# np.random.shuffle(rest_set)
# rest_set = rest_set[:30]
print('label len', len((extract_label)))


def shuffle_class(classes):
    np.random.shuffle(classes)
    tr_class = classes[:40]
    val_class = classes[40:50]
    test_class = classes[50:]
    return tr_class, val_class, test_class

def save_data(save_path, class_name, num, mod):
    path = '/mnt/data1/kinetics-skeleton'
    if mod == 'tr':
        data_path = os.path.join(path, 'train_data.npy')
        label_path = os.path.join(path, 'train_label.pkl')
    else :
        data_path = os.path.join(path, 'train_data.npy')
        label_path = os.path.join(path, 'train_label.pkl')
    origin_data = np.load(data_path)
    import pickle
    try:
        with open(label_path) as f:
            sample_name, origin_label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, origin_label = pickle.load(f, encoding='latin1')
    num_class = np.zeros(425)
    num_frame = [300 for i in range(len(origin_label))]


    data, label, frame = [], [], []
    for i in range(len(origin_label)):
        num_class[origin_label[i]] += 1
        if origin_label[i] in class_name:
            if num_class[origin_label[i]] > num:
                continue
            data.append(np.expand_dims(origin_data[i], axis=0))
            label.append(origin_label[i])
            frame.append(num_frame[i])

    print("data.shape", len(data))
    data = np.concatenate(data, 0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, mod + '_data.npy'), data)
    np.save(os.path.join(save_path, mod + '_label.npy'), label)
    np.save(os.path.join(save_path, mod + '_frame.npy'), frame)

nums = [30, 60, 100]
for i in nums:
    tr_class, val_class, test_class = shuffle_class(extract_label)
    print(tr_class)
    print(val_class)
    print(test_class)
    save_path = '/mnt/data1/kinetics-skeleton/select_motions/class_60_num_{}'.format(i)
    save_data(save_path, tr_class, i, 'tr')
    save_data(save_path, val_class, i, 'val')
    save_data(save_path, test_class, i, 'test')



