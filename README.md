# Chess diagram to FEN

Extract the FEN out of images of chess diagrams.

## Requirements

```shell
conda create -n chess_diagram_to_fen_env python=3.9
conda activate chess_diagram_to_fen_env
conda install pip

# Install torch for CPU, CUDA (Nvidia), or ROCm (AMD)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

conda install matplotlib scikit-image tqdm
pip install python-chess cairosvg pyfastnoisesimd

```

## Usage

```python
from PIL import Image
from chess_diagram_to_fen import get_fen

img = Image.open("your_image.jpg")
fen = get_fen(img=img, num_tries=10, return_cropped_img=False, auto_rotate=True)

print(fen)
```

Or use the demo program:
```shell
python chess_diagram_to_fen.py --shuffle_files --dir resources/test_images
```


## Train models yourself

#### Generate training data
Needs about **40 GB** disk space.
```shell
python main.py generate fen

# It is important to generate the fen data before
# the bbox data, since the bbox data generation
# relies on the fen training data

pip install gdown
./download_website_screenshots.sh
python main.py generate bbox

./download_lichess_games.sh
```

#### Review datasets (optional)

```shell
python main.py dataset bbox
python main.py dataset fen
python main.py dataset orientation
```

#### Train

```shell
python main.py train bbox
python main.py train fen
python main.py train orientation
```

#### Evaluate (optional)

```shell
python main.py eval fen
python main.py eval orientation
```

## Examples

### Successes

<img src="./resources/examples/success/success_1.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_2.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_3.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_4.jpg" width="600px" style="border-radius: 20px;">


### Failures

<img src="./resources/examples/failure/failure_1.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_2.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_3.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_4.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_5.jpg" width="600px" style="border-radius: 20px;">


