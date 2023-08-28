# Korean Handwriting Recognition AI
> Konkuk Univ. Senior, Multimedia Programming - Term project (Individual project)

# 1. Introduction
> "Can an individual's Korean handwriting be learned through an artificial neural network?"

Even when writing the same letters, each person's handwriting looks subtly different. The handwriting of people who are forced to write or who write while belching will leave different traces than usual, making it highly reliable as evidence. This uniqueness is what makes a signature unique. In important documents and exams, it is very important to check the handwriting for authenticity.

As it is used to prove identity, handwriting is unique, but if there are two different handwritings with only subtle differences, it is difficult for humans to distinguish the difference with the naked eye. Therefore, I wanted to implement an artificial intelligence model for Korean handwriting recognition.

## (1-1) Input data
- Korean Handwriting from 10 people 

<img src="/images/bsn_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/chw_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/kbj_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/kjh_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/ljh_2.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/lse_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/pjh_0.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/psm_1.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/shb_2.jpg" width="15%" height="15%" title="letter" alt="letter"></img>
<img src="/images/sws_2.jpg" width="15%" height="15%" title="letter" alt="letter"></img>

* The picture above is all written by different person.

## (1-2) Target
- 10 people (3 family members, 7 close friends)
    + BSN(Sona Bang), CHW(Howon Choi), KBJ(Beomjun Kim), KJH(Joonhyung Kwon), LJH(Jongho Lee), LSE(Seungeon Lee), PJH(Jonghyuk Park), PSM(Sangmoon Park), SHB(Suk Hyunbin), SWS(Woosub Shin)

## (1-3) Expected application field
- Handwriting verification of national examination and important documents.
- OCR (Optical Character Recognition)

# 2. Creating database
## 2-1. One-letter data
<p align="center">
  <img src="/images/img1.png" width="80%" height="80%" title="total loss" alt="total loss"></img>
</p>

### (1) imread
<img src="/images/1.png" width="35%" height="35%" title="letter" alt="letter"></img>

- Used template provided from "Ongle-leap", a font design company
- Included every combination of Korean letter

### (2) Unsharped mask
<img src="/images/2.png" width="35%" height="35%" title="letter" alt="letter"></img>

- Used unsharped mask to sharpen one's handwriting 

### (3) Grayscale transform
<img src="/images/3.png" width="35%" height="35%" title="letter" alt="letter"></img>

### (4) Histogram examine & Apply 1st threshold 
<img src="/images/4-2.png" width="35%" height="35%" title="letter" alt="letter"></img>
<img src="/images/4-1.png" width="35%" height="35%" title="letter" alt="letter"></img>

- The threshold is set through the Histogram, and the image is binarized based on the threshold. 
- (In this example, the threshold is set to 150 of 0 to 255)

### (5) Apply LPF
<img src="/images/5.png" width="35%" height="35%" title="letter" alt="letter"></img>

- In order to extract the position of the handwriting, it is necessary to smooth the handwriting through LPF so that the contour is exposed.
- Set the kernel size appropriately and apply LPF to binarized images through cv2.filter2D. 
- The smaller the Kernel size, the easier it is to detect smaller units such as vowels and consonants, and the larger the Kernel size, the easier it is to detect the contour of the letter itself.
- (Example applies 21x21 kernel)

### (6) Histogram examine & Apply 2nd threshold 
<img src="/images/6-1.png" width="35%" height="35%" title="letter" alt="letter"></img>
<img src="/images/6-2.png" width="35%" height="35%" title="letter" alt="letter"></img>

- Set the threshold through the histogram of the picture smoothed with LPF, then binarization is performed once again based on the threshold.
- (Example threshold value : 230)

### (7) Extract contour & coordinates, image crop
<img src="/images/7.png" width="35%" height="35%" title="letter" alt="letter"></img>

- A small outline that is not a letter, was not extracted.
- The x,y coordinates and w,h values were extracted from the extracted contour, and the coordinates were calculated again with a square so that the handwriting characteristics were not lost as much as possible because they had to be resized to 64x64 sizes later.
- Through the calculated coordinates, the image was cropped into a square shape.

### (8) Imwrite 
<img src="/images/8.png" width="50%" height="50%" title="letter" alt="letter"></img>

- 81 one-letter data per person
- total : 810 one-letter data were collected, 

## 2-2. Two-letter data
<img src="/images/bsn2_0.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn2_1.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn2_2.jpg" width="20%" height="20%" title="letter" alt="letter"></img>

- Two-letter data was made by combining one-letter data.
- In order to reduce the loss of feature information during resizing, the img_concat(img1, img2) function was created and used to connect the images, and forming them into squares.

* Acquired 6,480 two-letter data per target (81P2)

## 2-3. Three-letter data
<img src="/images/bsn3_0.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn3_1.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn3_2.jpg" width="20%" height="20%" title="letter" alt="letter"></img>

- Three-letter data was made by combining one-letter data and two-letter data.

* Acquired 7,980 three-letter data per target

## 2-4. Actual handwritten data from note-taking
<img src="/images/bsn4_1.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn4_2.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn4_3.jpg" width="20%" height="20%" title="letter" alt="letter"></img>
<img src="/images/bsn4_4.jpg" width="20%" height="20%" title="letter" alt="letter"></img>

- Data was acquired from the target's actual note taking.

* Acquired 30 actual handwritten data per target

# 3. 1st Result
- Train data : Test data = 9 : 1
- Used data = One-letter(81) + Two-letter(500) + Three-letter(500) + Actuall handwriting(30) 
- n.Epoch = 20, Batch size = 50, Learning rate = 0.01

- Layer configure

<img src="/images/result1.png" width="60%" height="60%" title="letter" alt="letter"></img>

- Result
  + Loss=0.7819, Accuracy=0.72
  + Result for test data=0.6503

<img src="/images/result2.png" width="60%" height="60%" title="letter" alt="letter"></img>

# 4. 2nd Result
- Train data : Test data = 9 : 1
- Used data = One-letter(81) + Two-letter(1000) + Three-letter(1300) + Actuall handwriting(30) 
- n.Epoch = 20, Batch size = 150, Learning rate = 0.04

- Layer configure

<img src="/images/result3.png" width="60%" height="60%" title="letter" alt="letter"></img>

- Result
  + Loss=0.0397, Accuracy=0.9908
  + Result for test data=0.9360
  
<img src="/images/result4.png" width="60%" height="60%" title="letter" alt="letter"></img>

# 5. Result analysis
Compared to the first result the second result were all significantly improved. I think the reason why the results have improved is as follows.

## (5-1) Increasing the data used for learning
- In the first attempt, 1,111 data were used per target, and in the second attempt, 2,411 data were used, which is more than double. 
- Since learning and evaluation were conducted through more data, more delicate characteristics of the target's handwriting could have been caputured.

## (5-2) Changes in the artificial neural network layer
- In the second attempt, the size of the convolution mask and the size of the pooling were modified.
- The size of the convolution mask was reduced to a smaller size than before to capture very fine handwriting characteristics
- Reduced the loss of handwriting characteristics in the learning process by reducing the pooling size.
- In addition, to contain more information about handwriting as parameters, the number of nodes in fully connected was increased from 128 to 256

## (5-3) Increase of batch size and learning rate
