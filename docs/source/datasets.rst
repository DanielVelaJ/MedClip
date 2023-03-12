Datasets
========

Medpix
------

A database of medical cases obtained from the `medpix online database 
<https://medpix.nlm.nih.gov/home>`_.
The database was scraped by `Luis R. Soenksen <soenksen@mit.edu>`_ and currently 
lives in Daniel's dropbox. 
It holds the following structure: 

.. code-block:: console

   .
   └── medpix
       ├── Images
       │   ├── 1.jpg
       │   ├── 2.jpg
       │   └── ...
       └── Dataset_MedPix_V1.xlsx

* The :file:`medpix/Images` folder contains all images in .jpg format.
* The :file:`medpix/Dataset_MedPix_V1.xlsx` contains an Excel file which refererences
  each image and includes the following columns. 

  * ID : This column indicates the name of the image corresponding to this row. 
      
  * Case_URL: The url to the medpix case from which this row was scraped. 
  * Subcase_URL
  * Diagnosis
  * Image_URL
  * Image_Title
  * Demographics
  * Caption: The caption corresponding to this image. 
  * Plane
  * Core_Modality
  * Full_Modality
  * ACR_Codes
  * Figure_Part
  * History
  * Exam
  * Findings: Findings for the whole case. 
  * Differential_Diagnosis
  * Case_Diagnosis
  * Diagnosis_By
  * Treatment_and_FollowUp
  * Discussion
  * Figure_Number
  * Topic_URL
  * Topic_Title
  * Disease_Discussion
  * Topic_ACR_Code
  * Location
  * Category
  * Keywords
  * References
  * External_Links
  * Author_URL
  * Authors
  * Licence_URL

.. warning::
    
   Notice that there can be more than one row/image per case. Which means several 
   rows may belong to the same case. Upon inspection of images that correspond to the seme 
   case (Case_URL) we found out that some of them are very simillar  or even the same image 
   with the exception of a small annotation or arrow. Also, the "Findings" section is shared
   across images from the same case. The "Caption" however  usually describes an individual
   image within a case. 

Mimics-cxr
----------
https://physionet.org/content/mimic-cxr/2.0.0/


By: Daniel Vela