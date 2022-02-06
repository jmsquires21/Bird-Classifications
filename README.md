
# **Classifying North American Birds**
### By: Jamie Squires


## Background & Overview

In recent years, image classification has become an integral aspect of machine learning. Within computer vision, there are 4 primary areas: image classification, localization, segmentation, and object detection. Among these areas, image classification is generally regarded as the most fundamental component of digital image analysis. Although image classification can be approached by both supervised and unsupervised learning methods, deep learning has emerged as the leader in image classification. In particular, the use of neural networks, specifically convolutional neural networks has risen in popular practice.

Image classification has many use cases in technology and in our daily lives. Whether it's identifying a stop sign for self-driving cars, medical image analysis, or identifying friends in a tagged photo on social media, the applications for computer vision are abundant. Despite the clear testament to technology, can we use computer vision and image classification to promote protection of the environment and deepen the appreciation for nature around us?

## Problem Statement

According to several studies, environmental knowledge can impact one's attitudes and subsequent behaviors. In one paper from [the U.S. National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839573/) "Loss of familiarity with the natural world, particularly in Western countries, is resulting in a loss of environmental knowledge, including the ability to identify even the most common species. Possessing at least basic animal and plant identification skills is often emphasized as a prerequisite for understanding and appreciating biodiversity." Essentially, the research shows that the more aware you are of different species around you, you're more likely to appreciate and understand subsequent species. This project aims to develop a model and tool to accurately identify North American birds in order to promote a fundamental knowledge of bird taxonomy, with the ultimate goal of empowering end users to further conservation efforts to protect birds.

* Can we use deep learning techniques to accurately predict bird images with their respective species?
* Can we create a useful tool to identify birds?


## Data & Methodology

We will leverage the [The Caltech-UCSD Birds 200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) for this project, which contains 11,788 images, with 200 categories of bird species found throughout North America. Each category contains roughly 60 images per species. We will focus on deep learning techniques and utilize neural networks to train our model.

### Success Metrics:
* Optimize for the highest accuracy score at the taxonomic order, family, and species level
* Create application to classify unseen bird images

### Methodology
1.	Explore the data, clean, transform images into arrays, condense and scale
2.	Run preliminary models, assess levels of accuracy
3.  In order to reduce the large number of classes (200), individual research was conducted to add in the taxonomic order and family into the dataset in order to improve the model's predictive power. The taxonomic hierarchy information came from the [The Cornell Lab of Ornithology](https://www.allaboutbirds.org/guide/browse/taxonomy).
4.	Build 3 separate neural networks to classify birds at the following levels:
    * Order (12 classes)
    * Family (35 classes)
    * Species (200 classes)
5. Iterate and experiment with the following deep learning techniques to optimize performance:
    * Image data augmentation (random rotation and random mirroring of the images in Tensorflow)
    * Transfer learning - leveraging a pretrained model
    * Exploring regularization techniques to reduce overfitting, early stopping, and batch normalization
6. Create an interactive tool in Streamlit to classify unseen images




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

* In the original dataset, we were given the species name along with a collection of images (shown on the right). You can see that each class of species shows only one bird in each photo, and they are generally scaled towards the center of the image. For this reason, we did not have to consider bounding boxes or additional image pre-processing. I used the [The Cornell Lab of Ornithology's Taxonomic Guide](https://www.allaboutbirds.org/guide/browse/taxonomy) to match each of the 200 species to their respective taxonomical order and family.


![Order bar chart](/Visualizations/order_hist.png)
* Once each species was mapped to their taxonomical order, one predominant class, Passeriformes, emerged. This comprises 67% of the image data. One interesting note here is that Passeriformes comprise more than half of all bird species.


![Family bar chart](/Visualizations/family_hist.png)
* At the family level, we can see that there are similar majority classes within the 35 families shown. The two largest groups (Parulidae and Passerellidae) both belong to the Passeriformes order. New World Warblers (Parulidae) and New World Sparrows (Passerellidae) make up 15% and 12% of the images.



# Modeling:


### Results
The final three neural network models were selected and compiled below:


| Class Level    | Number of Classes | Baseline Accuracy | Training Accuracy | Testing Accuracy |
|:--------------:|:-----------------:|:-----------------:|:-----------------:|:----------------:|
|      Order     | 12                | 67.0%             | 99.4%             | 87.6%            |
|     Family     | 35                | 15.2%             | 99.0%             | 64.0%            |
|     Species    | 200               | 0.5%              | 86.3%             | 30.5%            |

Highlights:
* All models significantly outperformed their baseline accuracies
* Data augmentation and transfer learning improved accuracy across the hierarchy


Drawbacks:
* All models were very overfit, despite regularization techniques implemented
* With only 60 images per species, we did not have enough data to improve the species and family model scores to a confident level.


Additional Notes:
* As expected, we see that the model that was run at the order level (12 classes) had the highest performance at 87.6%. This model was also the most flexible in that it responded positively to L2 regularization, batch normalization, and dropout layers.
* The family and species level models had a significantly lower accuracy level due to the increase in number of classes. Additionally, I found that these models reacted more unfavorably to regularization or further dense layers than the order model.

### Taxonomic Order Modeling Results
![Confusion Matrix Order](/Visualizations/order_confusion_matrix.png)

* Given that the Order model had the best scores and can be easily summarized in a confusion matrix,  we can see that Passeriformes performed the best at 96% accuracy. This is intuitive because Passeriformes was the majority class, making up 67% of the images.
* On the other hand, we see that Gaviiformes and Cuculiformes struggled with only 33% and 35% accuracy, respectively. They were both relatively mid to small classes.
* Gaviiformes' predictions were across the board, and there wasn't a dominant prediction class. However, Cuculiformes were classified as Passeriformes 60% of the time. Upon further manual inspection, they proved to be difficult to distinguish from various Passeriformes species.

# Conclusion:
### Learnings & Next Steps :
* Given the extensive number of classes in the original dataset (the 200 species), this was a big limiting factor in reaching reasonably high accuracy in our model. By aggregating the data using the taxonomic hierarchy structure, we were able to significantly improve our predictions and provide a tiered approach to accuracy.
* The models seemed to strongly benefit from data augmentation, with adding Tensorflow's layers.RandomRotation and layers.RandomFlip step in the Sequential() setup. Leveraging an existing pre-trained model and implementing transfer learning also had a huge impact on the accuracy levels. I used MobileNetV2. Early stopping, regularization, and adding dropout layers also benefitted the order model's performance.
* In the future, I'd like to get more images to improve the model. 60 images per class proved to not be enough data to accurately classify bird images at the species level. Augmenting the data through slight rotations and mirroring helped but it wasn't sufficient to handle the imbalanced classes. On that note, it would be especially helpful to gather more images in the smaller classes that performed poorly, such as the Gaviiformes and Cuculiformes.
* In order to further improve the accuracy of the model, I would pursue obtaining more computing power to run more advanced models. I used Google Colab Pro, but I'd like to find a more scalable solution with more GPU instances in the future. Google Cloud and AWS are two options I'd consider to eliminate bandwidth issues.





**Sources Cited:**


[The Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

[The Caltech-UCSD Birds-200-2011 Dataset Technical Report](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)

[The Cornell Lab of Ornithology- All About Birds](https://www.allaboutbirds.org/guide/browse/taxonomy)

[White RL, Eberstein K, Scott DM. Birds in the playground: Evaluating the effectiveness of an urban environmental education project in enhancing school children's awareness, knowledge and attitudes towards local wildlife. PLoS One. 2018;13(3):e0193993. Published 2018 Mar 6. doi:10.1371/journal.pone.0193993](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839573/)
