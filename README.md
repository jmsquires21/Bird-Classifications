
# **Classifying North American Birds**
### By: Jamie Squires


## Background & Overview

In recent years, image classification has become an integral aspect of machine learning. Within computer vision, there are focuses on image classification, localization, image segmentation, and object detection. Among these areas, image classification is generally regarded as the most fundamental piece of digital image analysis. Although image classification can be approached by both supervised and unsupervised learning methods, deep learning has emerged as the leader in image classification. In particular, the use of neural networks, specifically convolutional neural networks have become the go-to practice in this area.

Image classification has many use cases in technology and in our daily lives. Whether it's identifying a stop sign for self-driving cars, medical image analysis, or identifying you and your friends in a tagged photo on social media, the applications for computer vision are abundant. Despite the obvious testament to technology, can we use computer vision and image classification to promote protection of the environment and deepen the appreciation for natural life around us?

## Problem Statement

According to several studies, environmental knowledge can facilitate attitude formation and subsequent behaviors. In one paper from [the U.S. National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839573/) "Loss of familiarity with the natural world, particularly in Western countries, is resulting in a loss of environmental knowledge–including the ability to identify even the most common species. Possessing at least basic animal and plant identification skills is often emphasized as a prerequisite for understanding and appreciating biodiversity." Essentially, the research shows that the more aware you are of different species around you, you're more likely to appreciate and understand subsequent species. This project aims to develop a model and tool to accurately identify North American birds in order to promote a fundamental knowledge of bird taxonomy, with the ultimate goal of empowering end users to further conservation efforts to protect birds.

* Can we use deep learning techniques to accurately predict bird images with their respective species?
* Can we create a useful tool to identify birds?


## Data & Methodology

We will leverage the [The Caltech-UCSD Birds 200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) for this project, which contains 11,788 images, with 200 categories of bird species found throughout North America. Each category contains roughly 60 images per species. We will focus on deep learning techniques and utilize neural networks to train our model.

### Success Metrics:
* Optimize for the highest accuracy score at the taxonomic order, family, and species level
* Create Streamlit application to classify new bird images in real time

### Methodology
1.	Explore the data, clean, transform images into arrays, condense and scale
2.	Run preliminary models, assess levels of accuracy
3.  In order to reduce the large number of classes (200), individual research was conducted to add in the taxonomic order and family into the dataset in order to improve the model's predictive power. The taxonomic hierarchy information came from the [The Cornell Lab of Ornithology](https://www.allaboutbirds.org/guide/browse/taxonomy).
4.	We will build 3 separate neural networks to classify birds at the following levels:
    * Order (12 classes)
    * Family (35 classes)
    * Species (200 classes)
5. We will iterate and experiment with the following deep learning techniques to optimize performance:
  * Image data augmentation (random rotation and random mirroring of the images in Tensorflow)
  * Transfer learning - leveraging a pretrained model
  * Exploring regularization techniques to reduce overfitting, early stopping, and batch normalization
6. Lastly, we seek to create an interactive tool in Streamlit to classify images in real time.



### Data Dictionary


|       Column      |   Type  |              Dataset             | Calculated Field |                                Description                               |
|:-----------------:|:-------:|:--------------------------------:|:----------------:|:------------------------------------------------------------------------:|
| Image Id          | integer | class_map_taxonomy_directory.csv | 0                | Unique ID for each image in dataset, from original data source           |
| Image Name        | object  | class_map_taxonomy_directory.csv | 0                | Image name, directory for each image in it's respective folder           |
| Is Training Image | integer | class_map_taxonomy_directory.csv | 0                | Recommended train/test split from the original datasource (50/50 split)  |
| Class Id          | integer | class_map_taxonomy_directory.csv | 0                | Unique ID for each class (species) in dataset, from original data source |
| Class Name        | object  | class_map_taxonomy_directory.csv | 0                | Species name, directory folder for each class                            |
| Order             | object  | class_map_taxonomy_directory.csv | 1                | Taxonomical order, as denoted in Cornell Lab (cited below)               |
| Order_Num_Seq     | integer | class_map_taxonomy_directory.csv | 1                | Order ID, matching to the order from Cornell Lab (1-12)                  |
| Species           | object  | class_map_taxonomy_directory.csv | 1                | Taxonomical species group, as denoted in Cornell Lab (cited below)       |
| Species_Num_Seq   | integer | class_map_taxonomy_directory.csv | 1                | Species group ID, matching to the species group from Cornell Lab (1-35)  |
| Family            | object  | class_map_taxonomy_directory.csv | 1                | Taxonomical family, as denoted in Cornell Lab (cited below)              |
| Family_Num_Seq    | integer | class_map_taxonomy_directory.csv | 1                | Family ID, matching to the family from Cornell Lab (1-35)                |

**Data Sources**

[The Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

[The Cornell Lab of Ornithology - All About Birds](https://www.allaboutbirds.org/guide/browse/taxonomy)

# EDA:
![Hierarchy](/Visualizations/hierarchy_new.png)

* In the original dataset, we were given the species name along with a collection of images (shown on the right). You can see that each class of species shows only one bird in each photo, and they are generally scaled towards the center of the image. For this reason, we did not have to worry about bounding boxes or needed to additional image pre-processing.
* I used the [The Cornell Lab of Ornithology](https://www.allaboutbirds.org/guide/browse/taxonomy) to match each of the 200 species to their respective taxonomical order and family.


![Order bar chart](/Visualizations/order_hist.png)
* Once we mapped each species to their taxonomical order, we can see that there is one predominant class, Passeriformes. This comprises 67% of the image data. One interesting note here is that Passeriformes comprise more than half of all bird species, and they are generally known as perching birds.


![Family bar chart](/Visualizations/family_hist.png)
* At the family level, we can see that there are similar majority classes within the 35 families shown. The two largest groups (Parulidae and Passerellidae) both belong to the Passeriformes order.


![Top 10 Families](/Visualizations/top_family_chart.png)

* Similar to the two charts above, we can see that families within the Passeriformes order generally makes up the most common families.
* New World Warblers (Parulidae) and New World Sparrows (Passerellidae) make up 15% and 12% of the images. The only order outside of the Passeriformes are the Charadriiformes' family of Laridae (Gulls, Terns, and Skimmers), which make up nearly 8% of our images.




# Findings & Recommendations:


### Results
* After multiple iterations, the final models were selected and compiled below.
* As expected, we see that the model that was run at the order level (12 classes) had the highest performance at 87.6%. This model was the most flexible in that it responded positively to L2 regularization, batch normalization, and dropout layers. The family and species level models had a significantly lower accuracy level due to the increase in number of classes.

| Class Level    | Number of Classes | Baseline Accuracy | Training Accuracy | Testing Accuracy |
|:--------------:|:-----------------:|:-----------------:|:-----------------:|:----------------:|
|      Order     | 12                | 67.0%             | 99.4%             | 87.6%            |
|     Family     | 35                | 15.2%             | 98.7%             | 61.1%            |
|     Species    | 200               | 0.5%              | 86.3%             | 30.5%            |


### Closer Look at Order

![Confusion Matrix Order](/Visualizations/order_confusion_matrix.png)

* Given that the Order model had the best scores and is the easiest to view in a confusion matrix,  we can see that Passeriformes performed the best at 96% accuracy. This is intuitive because Passeriformes was the majority class, making up 67% of the images.
* On the other hand, we see that Gaviiformes and Cuculiformes struggled with only 33% and 35% accuracy, respectively. They were both relatively small classes.
*For Gaviiformes, we see that the predictions were across the board, and there wasn't a dominant prediction class. However, Cuculiformes were classified as Passeriformes 60% of the time. Upon further manual inspection, they were difficult to distinguish from various Passeriformes species.

### Learnings:
* Given the extensive number of classes in the original dataset (the 200 species), this was a big limiting factor in reaching reasonably high accuracy in our model. By aggregating the data using the taxonomic hierarchy structure, this proved to be a good solution to the issue of too many classes. It was also interesting to see how model performance changed once we reduced the number of classes to orders and families.
* The models seemed to benefit from data augmentation, with adding Tensorflow's layers.RandomRotation and layers.RandomFlip step in the Sequential() setup. Leveraging an existing pre-trained model and implementing transfer learning also had a huge impact on the accuracy levels. I used MobileNet V2.
* Early stopping, regularization, and adding dropout layers also benefitted my model's performance.
* In order to further improve the accuracy of our neural network, I would pursue obtaining more computing power to run more advanced models. I used Google Colab Pro to run the models above, but I would've ideally had access to AWS. I was unable to obtain the adequate number of GPU instances, but given more time, I'd like to find some other resources.


### Next Steps:
* Address the issue of imbalanced classes
* I'd like to continue to develop more advanced models by gaining access to AWS or another hosted server to eliminate bandwidth issues









**Sources Cited:**


[The Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

[The Caltech-UCSD Birds-200-2011 Dataset Technical Report](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)

[The Cornell Lab of Ornithology- All About Birds](https://www.allaboutbirds.org/guide/browse/taxonomy)

[White RL, Eberstein K, Scott DM. Birds in the playground: Evaluating the effectiveness of an urban environmental education project in enhancing school children's awareness, knowledge and attitudes towards local wildlife. PLoS One. 2018;13(3):e0193993. Published 2018 Mar 6. doi:10.1371/journal.pone.0193993](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839573/)
