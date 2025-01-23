# Auto-annotation Model
 
## Overview  
This repository contains an **Auto-annotation Model** using a pre-trained model that is yolov8n that detects and classifies the MRP area in images of cement bags, as well as other specified image types. The model automates the annotation process, saving significant manual effort. It supports both MRP and print vision auto-annotation tasks.  

The auto-annotation process leverages state-of-the-art machine learning techniques for accurate and efficient annotations, providing annotated images ready for use in various workflows.  

---

## Requirements  
- **Python**: Version `3.12.0` (recommended)  
  - [Download Python 3.12.0](https://www.python.org/downloads/release/python-3120/)  
- **Ultralytics**  
  - Clone the [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)  
  - Create a virtual environment:  
    ```bash
    python -m venv env
    ```  
  - Activate the environment and install Ultralytics:  
    ```bash
    pip install ultralytics
    ```  
- **VS Code**  
  - If not installed, [Download Visual Studio Code](https://code.visualstudio.com/)  

- **LabelImg**  
  - Install LabelImg using this [GitHub Repository](https://github.com/tzutalin/labelImg)  

---

## Installation Instructions  

To use the trained model for **MRP** and **Print Vision Auto-annotation**, follow these steps:  
1. Clone this repository into the same directory where Ultralytics (virtual environment) was downloaded:  
    ```bash
    git clone <https://github.com/Sakshi13t/Auto-annotation>
    cd <MRP> for MRP detection
    cd <Printvision> for Character detection
    ```  

2. Open `infer_model.py`, the script responsible for performing annotations.  

3. Update the path to the image directory in the script:  
    ```python
     source = "path/to/your/image/directory"
    ```  

4. Run the script to generate annotated images:  
    ```bash
    python infer_model.py
    ```  

5. The annotated images will be saved in the output directory specified in the script.  

---

## Notes  
- The model is pre-trained on a specific dataset. If the results are not satisfactory for your use case, consider fine-tuning the model or contact the repository owner for assistance.  
- Fine-tuning may involve additional data preparation and training steps to adapt the model to your specific image types.  

---

## Example Output  

Annotated images with detected regions of interest (e.g., MRP area) are generated and saved automatically. Here's an example of the expected output:  
![Annotated Example-1](https://github.com/Sakshi13t/Auto-annotation/blob/main/MRP/notok_classification.jpg)  
![Annotated Example-2](https://github.com/Sakshi13t/Auto-annotation/blob/main/MRP/ok_classification.jpg)  
![Annotated Example-3](https://github.com/Sakshi13t/Auto-annotation/blob/main/Printvision/Img_1.jpg) 
![Annotated Example-4](https://github.com/Sakshi13t/Auto-annotation/blob/main/Printvision/Image_1727.jpg) 

--- 

## Contact  
For further queries or assistance, feel free to open an issue or contact the repository owner.  
