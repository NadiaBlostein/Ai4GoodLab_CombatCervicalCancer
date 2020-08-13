# BACKGROUND 

Cervical cancer is the fourth most common cancer in women worldwide. However, if precancerous cells are detected on time, cervical cancer is preventable 90% of the time. In spite of this, in 2018, 311 000 women died from this disease. There are several explanations for this cancer’s insidiously high mortality rate. A lack of proper sexual health education prevents people from taking the necessary precautions to prevent the spread of HPV, a sexually transmitted infection (STI) that causes 90% of cases of cervical cancer. Furthermore, Gardasil vaccines, which protect people against many strains of HPV, can be expensive and not easily accessible. Another crucial pre-emptive measure to avoid this cancer is the screening for precancerous cells in the cervix. This, however, requires regular access to the proper healthcare services. Finally, the frequently inappropriate choice of treatment for cervical cancer may be extremely dangerous for the patient. Indeed, after precancerous cells are detected, it is pivotal to adapt the treatment to each individual patient’s cervix position. In light of this, our group sought to build two tentative solutions to both the issues of diagnosis and treatment of this cancer. (1) We built a cervical cancer risk prediction score which gages the urgency with which somebody should be subject to a PAP test (precancerous cell screening test). We believe that this is a tool that could be useful for women with poor access to healthcare services. (2) We built a cervix position type classifier, trained on cancer-naive subjects, which would prospectively serve the purpose of helping physician properly customize cervical cancer treatment to each patient.

# DATASET 1: Risk Prediction Score

We built a cervical cancer risk prediction score based off of data collected from 858 patients who underwent a gynecological exam in Caracas, Venezuela, including 55 subjects with cervical cancer. We found this data on Kaggle: https://www.kaggle.com/loveall/cervical-cancer-risk-classification.

*Feature correlation matrix.* The feature correlation matrix seems to indicate that our sample size is too small for any real correlations to crop up. However, there was, as expected, a slightly stronger correlation between number of pregnancies and age as well as years with an IUD and age, which makes the dataset seem slightly more robust (see *risk_score_SVC.ipynb*).

*Input.* The features that we input into our model were each subject’s past contraceptive, sexual and STD medical history (see *risk_score_kaggle_data.csv*).

*Model.* After testing several classifiers, we found that a linear support vector classifier performed the best with an accuracy of 66%. To compensate for the uneven class distribution, we bootstrapped the training data and examined the model’s balanced accuracy on the test set (see *risk_score_SVC.ipynb*).

*Output.* The model outputs an integer value between 0 (meaning no risk) and 4 (meaning high risk) as a subject’s cervical cancer risk prediction score (see *risk_score_SVC.ipynb*).

# DATASET 2: Cervix Position Classification

Our cervix position classifier was trained on another Kaggle dataset: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/notebooks. This dataset consisted of over 8000 images whose labels belonged to one out of three classes: Type 1, Type 2, Type 3.

*Input.* 2D images of cervix in .jpg format, each of which was either labelled as Type 1, Type 2 or Type 3. The data was subdivided into train, test and validation sets. Image preprocessing include pixel value within-channel normalization and image resizing (see *sample_train_cervix_shapes* for a 60 image examples).

*Model.* ResNet 18 was the CNN that performed the best with a test accuracy of 70% (see *cervix_position_classifier_CNN.ipynb*).

*Output.* For each input image, the model outputs one out of three labels, each belonging to a specific cervix position type (see *cervix_position_classifier_CNN.ipynb*). 

*Image segmentation.* As a prospective preprocessing step, we segmented the cervix in several test images. A constrast split segmentation mask was used to binarize intensities in order to distinguish between dark and light components of the image. In the future, we aim to incorporate this segmentation algorithm into our preprocessing pipeline (see *imageSegmentation-checkpoint* for the segmentation code and *scitkit_image_exploration.ipynb* for our exploration of the .jpg data).

# Future Works.

To improve our cervical cancer risk prediction score, we would like to:\
  (a) add features to the dataset, such as socioeconomic status and education;\
  (b) tackle the problem of data accessibility by consensually collecting and anonymizing the data of our app users and directly feeding it into the model.
  
To improve our cervix position classifier, we would like to:\
  (a) apply AWS Hyperparameter Optimization (HPO) to our model;\
  (b) incorporate image segmentation into our image preprocessing pipeline;\
  (c) train our model on the additional data that was available on Kaggle;\
  (d) manually double-check our ground-truth cervix labels with an expert.
