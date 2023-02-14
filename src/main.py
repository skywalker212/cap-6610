import random
import matplotlib.pyplot as plt
from loader import MnistDataloader
from os.path  import join

input_path = './data'
training_images_filepath = join(input_path, 'train-images/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 'test-images/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 'test-labels/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 12);        
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
total_train_images = len(y_train)
total_test_images = len(y_test)


#
# Show 10 random training and 5 random test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, total_train_images)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, total_test_images)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)