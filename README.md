### 1. Requirements Installation

Make sure you have Python installed on your system. Then, run the following command to install the necessary dependencies:

```
pip install -r requirements.txt
```

Download the model from [this link](https://drive.google.com/file/d/1LPn2XUDiY3Mtbo78BVJxQX0TbHr78a-2/view?usp=sharing) and paste it into the repository folder.

### 2. Running Inference

```
python inference.py
```

Replace `"video_path"` with the path to your video file in `inference.py` to run inference on any other video file.


### Important

The inference process will be faster if your system has a GPU and CUDA support. If your system does not have a GPU or CUDA, 
the inference process might be slower. Therefore, it is recommended to use a system with a GPU for optimal performance.
