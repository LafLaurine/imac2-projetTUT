
  <!-- PROJECT LOGO -->

  # Deepfake detection tools integrated to [WeVerify InVID plugin](https://www.invid-project.eu/tools-and-services/invid-verification-plugin/)
![InVIDPlugin](https://is2-ssl.mzstatic.com/image/thumb/Purple118/v4/62/f0/c7/62f0c79d-401a-4919-ef8c-3c934c44b51d/source/256x256bb.jpg)

  <!-- TABLE OF CONTENTS -->
  ## Table of Contents

  * [About the Project](#about-the-project)
    * [Built With](#built-with)
  * [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
  * [Services](#services)  
  * [Installation](#installation)
    * [Standalone](#standalone)
    * [Plugin](#plugin) 
  * [Usage](#usage-of-the-services)
    * [Standalone](#standalone)
    * [Plugin](#plugin) 
  * [Authors](#authors)
  * [Acknowledgements](#acknowledgements)
  * [License](#license)


  <!-- ABOUT THE PROJECT -->
  ## About The Project

  The purpose of this project is to detect deepfakes videos thanks to several methods that already exist ([MesoNet](https://github.com/DariusAf/MesoNet), [CapsuleForensics](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)) and include it to the [WeVerify InVID plugin](https://github.com/AFP-Medialab/we-verify-app) project.

  This repository includes sources that can be run with the help of Docker tools, to train neural networks and/or use them to detect deepfakes.

  It can be used as a standalone or with the WeVerify InVID plugin.


  ### Built With

  * [Python](https://www.python.org/) - Python is a programming language that lets you work quickly and integrate systems more effectively
  * [Docker](https://www.docker.com/) - Docker is an open source software allowing to launch applications in software containers
  * [docker-compose](https://docs.docker.com/compose/) - A tool for defining and running multi-container Docker applications
  * [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Open-source web development framework in Python
  * [Maven](http://maven.apache.org/) - Apache Maven is a software project management and comprehension tool. Based on the concept of a project object model (POM), Maven can manage a project's build, reporting and documentation from a central piece of information 
  * [SpringBoot](https://spring.io/projects/spring-boot) - Spring Boot makes it easy to create stand-alone, production-grade Spring based Applications that you can "just run"
  * [React](https://en.reactjs.org/) - A JavaScript library for building user interfaces

  <!-- GETTING STARTED -->
  ## Getting Started

  To get a local copy running, follow these steps.

  ### Prerequisites

  **This project has been tested on Linux Ubuntu 18.04 and 19.04. It should run normally on Windows.**

  #### Docker

  Make sure Docker is intalled on your environment.
  ```sh
  docker --version
  ```
  If not, install it by following these instructions : 
  * For Ubuntun 18.04 : [How to install Docker on Ubuntu 18.04](https://phoenixnap.com/kb/how-to-install-docker-on-ubuntu-18-04)
  * For Windows : [Get started with Docker for Windows](https://docs.docker.com/docker-for-windows/)

  #### Docker-compose
  * For Ubuntu : 
  You will also need to install the docker-compose tool.
  ```sh
  sudo curl -L https://github.com/docker/compose/releases/download/1.18.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  docker-compose --version
  ```
  You can get the latest version of docker-compose in the documentation : [Install docker-compose](https://docs.docker.com/v17.09/compose/install/#master-builds)

  * For Windows : [Install Docker compose](https://docs.docker.com/compose/install/)

  ##### Link to Docker images
  *Unfortunatly because of the current situation, docker images couldn't be updated*

  [DockerHub](https://hub.docker.com/repository/registry-1.docker.io/laflaurine/afp-df_api/tags?page=1)

  #### JDK 1.8  or above
  * For Ubuntu : [https://tecadmin.net/install-oracle-java-8-ubuntu-via-ppa/](https://tecadmin.net/install-oracle-java-8-ubuntu-via-ppa/)
  * For Windows : [https://www.oracle.com/java/technologies/javase-jdk8-downloads.html](https://www.oracle.com/java/technologies/javase-jdk8-downloads.html)

  #### Maven 3.2 or above
  * For Ubuntu : [https://linuxize.com/post/how-to-install-apache-maven-on-ubuntu-18-04/](https://linuxize.com/post/how-to-install-apache-maven-on-ubuntu-18-04/)
  * For Windows : [https://maven.apache.org/guides/getting-started/windows-prerequisites.html](https://maven.apache.org/guides/getting-started/windows-prerequisites.html)

  ### Plugin

  #### NodeJS
  You need to install [NodeJs](https://nodejs.org/en/) and run `npm install` on this folder on your first download to get dependencies.

  ## Services

  Quick overview of all implemented services. Each services works on his own. 

  - **extraction_dir** : extract faces from multiple videos that are in a directory / *Works on : [localhost:8080/extraction/dir](http://localhost:8080/extraction/dir)*
  - **extraction_video** : extract faces from a video / *Works on [localhost:8080/extraction/video](http://localhost:8080/extraction/video)*
  - **mesonet_test** : check if MesoNet neural network is well trained / *Works on [localhost:8080/mesonet_test](http://localhost:8080/mesonet_test)*
  - **mesonet_analyse** : after faces extracted, check if a video is a deepfake or not  / *Works on [localhost:8080/mesonet_analyse](http://localhost:8080/mesonet_analyse)*
  - **mesonet_training** : train MesoNet neural network /  *Works on [localhost:8080/mesonet_training](http://localhost:8080/mesonet_training)*
  - **capsule_forensics_test** : check if CapsuleForensics neural network is well trained  / *Works on [localhost:8080/capsule_forensics_test](http://localhost:8080/capsule_forensics_test)*
  - **capsule_forensics_analyse** : after faces extracted, check if a video is a deepfake or not /  *Works on [localhost:8080/capsule_forensics_analyse](http://localhost:8080/capsule_forensics_analyse)*
  - **capsule_forensics_training** : train CapsuleForensics neural network  /  *Works on [localhost:8080/capsule_forensics_training](capsule_forensics_training)*

  If you go to **fakedetection/src/main/docker/extraction** you'll see a file that is named *display_faces_capture.py*, it doesn't have a service because his only role is showcase. You will need a webcam in order to use this application.
  You can run it with : 

  ```sh
  python3 display_faces_capture.py --method DNN or python3 display_faces_capture.py --method DNN_TRACKING
  ```
  This application allows you to see how we can extract faces : when the application is running, you'll can use **r** and **l** on  your keyboard to see the landmarks and the rectangle of your face, which are used to extract faces.

  ## Installation

  ### Standalone

  1. Clone the repository
  ```sh
  git clone https://github.com/laflaurine/imac2-projetTUT.git
  cd imac2-projetTUT
  ```
  2. Move into the **fakedetection/** directory where you can find the spring boot project
  ```sh
  cd fakedetection
  ```
  3. Run the project by using the following command:
  ```sh
  sudo mvn compile
  sudo mvn package
  sudo mvn install
  java -jar target/fakedetection-0.0.1-SNAPSHOT.jar
  ```
  4. Move into the **fakedetection/src/main/docker** directory where you can find the docker-compose.yml file
  ```sh
  cd fakedetection/src/main/docker
  ```
  5. You can either run all services at once :
  ```sh
  sudo docker-compose up
  ```
  Or run them one by one (if you want to run it with your own option and not with the default) : 
  ```sh
  sudo docker-compose up name_of_the_service
  ```

  6. The service should be running on their own host, you'll see in the console which port is running or you can copy the host from above. Example : go to http://localhost:8080/extraction/dir

  _For detailed information on how to run each service, please refer to the example [usages](#usage-of-the-services) below._

  ### Plugin
  1. Clone the repository
  ```sh
  git clone https://github.com/laflaurine/imac2-projetTUT.git
  cd imac2-projetTUT
  ```

  2. You will need a  `.env`  file containing :
  ```
  REACT_APP_ELK_URL=<ELK-URL>/twinttweets/_search
  REACT_APP_TWINT_WRAPPER_URL=<TWINT-WRAPPER-URL>
  REACT_APP_FACEBOOK_APP_ID=<REACT_ID>
  REACT_APP_TRANSLATION_GITHUB=https://raw.githubusercontent.com/AFP-Medialab/InVID-Translations/react/
  REACT_APP_KEYFRAME_TOKEN=<yourKeyframeToken>
  REACT_APP_MY_WEB_HOOK_URL=<yourSlackAppUrlHook>
  REACT_APP_GOOGLE_ANALYTICS_KEY=<yourGoogleAnaliticsToken>
  REACT_APP_MAP_TOKEN=<MAP_TOKEN>
  REACT_APP_AUTH_BASE_URL=<TWINT-WRAPPER-URL>
  ```
  3. Run `npm run build` to build the app for production to the  `build`  folder.

  4. Run: `npm start` in order to run the app in the development mode.  This will run on port  [3000](http://localhost:3000/).

  5.  Use the extension
  **For Chrome :**
  -   In chrome menu go to  `More tools`  then click  `Extentions`
  -   Activate the  `Developer mode`  toggle
  -   The click the  `Load Unpacked`  button
  -   Select the  `dev`  or  `build`  file you generated earlier.

      **For Firefox :**
  - In firefox menu click on  `Add-ons`
  -   Then click on the gear button  `⚙⌄`
  -   Then click on  `Debug Add-ons`
  -   Then click on  `Load Temporary Add-on...`
  -   Select the  `manifest.json`  in the  `dev`  or  `build`  file you generated earlier.

  ####  Detailed plugin functionality can be found [on the WeVerify - InVID github](https://github.com/AFP-Medialab/we-verify-app)

  Unfortunalty, you still have to follow the [Usage of the services](#usage-of-the-services) section, as the back-end doesn't work fully alone. 

  <!-- USAGE EXAMPLES -->
  ## Usage of the services

  **Arguments**

  The arguments used by each services are declared by default in the `fakedetection/src/main/docker/.env` file.

  You would probably set your own paths when running the services to better fit your working environment. You can modify any of these variables by declaring them when running the command (as you can see below).

  *Default arguments:*
  ```sh
  input_path=input/video.mp4 # Path to the video you want to analyze
  video_download=False # Boolean value. Is needed True if you want to download a video  
  video_url =https://www.youtube.com/watch?v=gLoI9hAX9dw # Video URL that you want to download
  name_video_downloaded=video # Name of the video you want to give to the downloaded one
  input_path_dir=./input/ # Path to the folder in which needed normalized images are stored
  output_path=output # Path to the folder in which needed normalized images will be stored
  method_detection=DNN # Can either be DNN or DNN_TRACKING
  start_frame=0 # Frame at which to start extracton
  step_frame=25 # Extract faces every ... frames
  end_frame=200 # Frame at which to end extraction
  max_frame=50 # Max of frames to extract
  are_warped=True # Faces will be aligned on the basis of eyes and mouth.
  are_culled=False # Faces will be culled according to out-of-bounds landmarks.
  are_saved_landmarks=False # Facial landmarks will be saved along with the corresponding face.
  is_saved_rectangle=False # IF NOT WARPED: Rectangle from face detection will be drawn in output image.
  mesonet_classifier=MESO4_DF # Can be Meso4_DF or Meso4_F2F or MesoInception_DF or MesoInception_F2F
  number_epochs=3 # Number of epochs
  batch_size=8 # Number of images in each batch
  path_to_dataset=dataAnalyse/out # Path to the analyse dataset
  train_dataset=data # Path to the training dataset
  capsule_forensics_classifier=BINARY_DF # Can be BINARY_DF or BINARY_F2F or BINARY_FACESWAP or MULTICLASS
  step_save_checkpoint=5 # Step at which to save temporary weights
  epoch_resume=1 # Which epoch to resume (starting over if 0)
  version_weights=2 # Version of the weights to load (has to be > 0)
  ```

  **Extraction video**

  *This service is a useful tool allowing you to extract normalized images that are needed to launch any other services.*
  *You can either extract faces from a video that you already have in your computer or download one from YouTube or Facebook thanks to youtube_dl.*

  *Default arguments:*
  ```sh
  input_path=input/video.mp4
  method_detection=DNN
  are_saved_landmarks=True
  video_download=False
  output_path=output
  name_video_downloaded=video
  are_warped=True
  are_culled=False
  is_saved_rectangle=False
  start_frame=0
  step_frame=25
  end_frame=200
  max_frame=50
  ```

  *Extract a video that you own*

  1. Make sure to put the video you want to analyze in a local folder (it's better if it's at the project root).

  2. *extraction_video* service : 
  The input video must be named "video.mp4"
  ```
  sudo video_download=False input_path=your_path_to_the_video/video.mp4 output_path=your_path_to_your_output_folder docker-compose up extraction_video
  ```

  *Extract a video from youtube*
  1. Make sure to copy the URL from the video that you want. Video name must be "video.mp4"

  2. Run the *extraction_video* service with the following command :
  ```
  sudo video_download=True video_url=your_video_url name_video_downloaded=video.mp4 output_path=your_path_to_your_output_folder docker-compose up extraction_video
  ```
  You can add other arguments following the same model as above.

  **Extraction directory**

  *This service is a useful tool allowing you to extract all normalized images stored in a directory.*

  *Default arguments:*
  ```sh
  input_path_dir=input_dir
  method_detection=DNN
  are_saved_landmarks=True
  output_path=output
  are_warped=True
  are_culled=False
  is_saved_rectangle=False
  start_frame=0
  step_frame=25
  end_frame=200
  max_frame=50
  ```

  1. Make sure that you have a local folder filled with videos that you want to extract faces. They must be in separated folder. Example : videos/video1; videos/video2

  2. Run the *extraction_dir* service with the following command :
  ```
  sudo input_path_dir=your_path_to_the_directory output_path=your_path_to_your_output_folder docker-compose up extraction_dir
  ```
  You can add other arguments following the same model as above.
  
  **MesoNet training**

  *This service is a useful tool allowing you to train MesoNet models*

  Default arguments : 
  ```sh
  mesonet_classifier=MESO4_DF
  train_dataset=data
  batch_size=8
  number_epochs=3
  step_save_checkpoint=5
  ```

  *Once you have extract the needed inputs, you can train the MesoNet method with them.*

  1. Make sure your images in the output folder are saved into two subfolders : `df` (for the images extracted from deepfake videos) and `real` (for the images extracted from real videos).
  Example : `train_dataset/df/your_deepfake_images.PNG` and `train_dataset/real/your_real_images.PNG`

  2. Run the *mesonet_training* service with the following command :
  ```
  sudo train_dataset=your_path docker-compose up mesonet_training
  ```
  You can add other arguments following the same model as above.

  **MesoNet test**

  *This service is a useful tool allowing you to test MesoNet models*

  Default arguments : 
  ```sh
  train_dataset=data
  name_classifier=MESO4_DF
  batch_size=8
  number_epochs=3
  ```

  1.  Make sure your images in the output folder are saved into two subfolders : `train` and `validation`.
  Example : `train_dataset/train/your_deepfake_images.PNG` and `train_dataset/validation/your_real_images.PNG`

  2. Run the *mesonet_test* service with the following command :
  ```
  sudo train_dataset=your_path docker-compose up mesonet_test
  ```
  You can add other arguments following the same model as above.

  **MesoNet analyse**

  *This service is a useful tool allowing you to analyse whether it's a deepfake or not with MesoNet method*

  Default arguments : 
  ```sh
  path_to_dataset=dataAnalyse/out
  name_classifier=MESO4_DF
  batch_size=8
  ```

  1. Make sure your images in the output folder are saved into subfolders.
  Example : `path_to_dataset/subfolder/images.PNG`

  2. Run the *mesonet_analyse* service with the following command :
  ```
  sudo path_to_dataset=your_path docker-compose up mesonet_analyse
  ```
  You can add other arguments following the same model as above.


  **CapsuleForensics training**

  *This service is a useful tool allowing you to train CapsuleForensics models*

  Default arguments
  ```sh
  capsule_forensics_classifier=BINARY_DF
  train_dataset=data
  batch_size=8
  number_epochs=3
  epoch_resume=1
  step_save_checkpoint=5
  ```

  1. Make sure your images in the output folder are saved into subfolders.
  Example : `train_dataset/subfolder/your_images.PNG`

  2. Run the *capsule_forensics_training* service with the following command :
  ```
  sudo train_dataset=your_path docker-compose up capsule_forensics_training
  ```
  You can add other arguments following the same model as above.
  
  **CapsuleForensics test**

  *This service is a useful tool allowing you to test CapsuleForensics models*

  Default arguments : 
  ```sh
  capsule_forensics_classifier=BINARY_DF
  train_dataset=data
  batch_size=8
  number_epochs=3
  version_weights=2
  ```

  1. Make sure your images in the output folder are saved into subfolders.
  Example : `train_dataset/subfolder/your_images.PNG`

  2. Run the *capsule_forensics_test* service with the following command :
  ```
  sudo train_dataset=your_path docker-compose up capsule_forensics_test
  ```
  You can add other arguments following the same model as above.

  **CapsuleForensics analyse**

  *This service is a useful tool allowing you to analyse whether it's a deepfake or not with CapsuleForensics method*

  Default arguments : 
  ```sh
  capsule_forensics_classifier=BINARY_DF
  path_to_dataset=dataAnalyse/out
  batch_size=8
  version_weights=2
  ```

  1. Make sure your images in the output folder are saved into subfolders.
  Example : `path_to_dataset/subfolder/your_images.PNG`

  2. Run the *capsule_forensics_analyse* service with the following command :
  ```
  sudo path_to_dataset=your_path docker-compose up capsule_forensics_analyse
  ```
  You can add other arguments following the same model as above.

  <!-- AUTHORS -->
  ## Authors

  * **Laurine Lafontaine** - [Github](https://github.com/laflaurine)
  * **Pierre Thiel** - [Github](https://github.com/piptouque)
  * **Manon Sgro'** - [Github](https://github.com/ManonSgro)

  <!-- ACKNOWLEDGEMENTS -->
  ## Acknowledgements

  * **AFP medialab** - [Website](https://www.afp.com/fr/lagence/medialab)
  * **Vincent Nozick** - [Website](http://www-igm.univ-mlv.fr/~vnozick/)
  * **Denis Teyssou** - [Github](https://github.com/AFPMedialab)
  * **Bertrand Goupil** - [Github](https://github.com/AFPMedialab)
  * **IMAC engineering school** - [Website](https://www.ingenieur-imac.fr/)

  <!-- LICENSE -->
  ## License
  The project is released under MIT, see [LICENSE](https://github.com/LafLaurine/imac2-projetTUT/blob/weverify/LICENSE).
