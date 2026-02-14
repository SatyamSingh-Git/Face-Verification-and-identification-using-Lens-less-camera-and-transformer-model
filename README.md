# Privacy-Preserving Face Recognition and Verification with Lensless Camera
This is the official implementation of our IEEE TBIOM 2024 paper titled [Privacy-Preserving Face Recognition and Verification With Lensless Camera](https://ieeexplore.ieee.org/document/10793399).

The code supports face recognition and verification directly from lensless sensor measurements, without reconstructing images, using the proposed Multi-resolution DCT Subband Representation.

<div align="center">
  <img src="overview_TBIOM_2024.jpg" alt="Overview" width="550"/>
</div>

## Prerequisites
The code is tested on Linux with the following prerequisites:

1. Python 3.10
2. PyTorch 1.11.0 (CUDA 11.3)
3. Numpy 1.26.4

## Installation

- Clone this repository
   ```bash
   git clone https://github.com/engrchrishenry/lensless_face_recognition.git
   cd lensless_face_recognition
   ```

- Create conda environment
   ```bash
   conda create --name lfc python=3.10
   conda activate lfc
   ```

- Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

### Option 1: Use Pre-computed Lensless Data
You may download the pre-computed lensless dataset required for training or testing the system [here](https://mailmissouri-my.sharepoint.com/:u:/g/personal/chffn_umsystem_edu/IQBgLURyOuKgSrwfl8fLn8ipAY6Ikc-va09tctmaHQaVGcY?e=1Xz4Bo). 

Unzip the data into the parent directory of this repository.

### Option 2: Generate Dataset from Scratch

Download the [**FlatCam Face Dataset**](https://computationalimaging.rice.edu/databases/flatcam-face-dataset/) by Rice University. Particularly, download the "Raw captures" ("fc_captures.tar.gz") which contains raw Flatcam sensor measurements.

> **Note:** This work uses the **FlatCam Face Dataset**. Users are responsible for complying with the dataset’s license and usage terms as specified by the original authors.

1. Split the data into train/test.
   ```bash
   cd fc_data_process/
   python prep_data_recog_complete.py --data_path <path_to_input_data> --out_path <path_to_splitted_data>"
   ```
2. [Optional] Run the following to generate pseudo-random noise locations. Skip this step to use pre-computed noise locations in [noise_locations](https://github.com/engrchrishenry/lensless_face_recognition/tree/main/data/noise_locations) consistent with the paper.
   ```bash
   python generate_noise_locations.py --loc_per_pixel 10
   ```
3. Generate the lensless dataset containing the proposed Multi-resolution DCT Subband Representation. May consume a couple of minutes to process.
   ```bash
   python process_raw_fc_meas.py --data_path <path_to_splitted_data/train> --out_path <path_to_output_folder/train>
   python process_raw_fc_meas.py --data_path <path_to_splitted_data/test> --out_path <path_to_output_folder/test>
   ```
   Output folders:

   - ymdct_npy: Contains the proposed Multi-resolution DCT Subband Representation (ymdct) (.npy).
   - ymdct_noisy_npy: Contains the proposed Multi-resolution DCT Subband Representation with pseudo-random noise (.npy).
   - dct_vis: Contains visualization of DCT of the sensor measurement (.jpg).
   - meas_vis: Contains visualization of the resized sensor measurement (.jpg).
4. [Optional] Generate verification pairs for testing. Skip this step to use pairs consistent with our paper ([pairs_verification.txt](https://github.com/engrchrishenry/lensless_face_recognition/blob/main/data/verification_pairs.txt)).
   ```bash
   python generate_verification_pairs.py --data_path "lensless_data/test/ymdct_npy" --output_file "pairs.txt" --num_of_pairs 10000
   ```

## Training
To train the network:
```bash
python train.py --train_data "lensless_data/train/ymdct_npy" --test_data "lensless_data/test/ymdct_npy" --batch_size 64 --lr 0.05 --num_epochs 100
```
## Testing

Download the [pre-trained weights](https://mailmissouri-my.sharepoint.com/:u:/g/personal/chffn_umsystem_edu/IQAI5HfkPTPnT4zYokmAKaLCAUGn34FcO1CFXHa0eA3iARw?e=nkmEhg) or train the network from scratch and copy the weights to the [weights](https://github.com/engrchrishenry/lensless_face_recognition/tree/main/weights) folder.

- ### Face Recognition (ymdct)
   ```bash
   python test_face_recognition.py --test_data "lensless_data/test/ymdct_npy" --weights "weights/pretrained_weights.pth" --batch_size 64
   ```
- ### Face Recognition (ymdct with pseudo-random noise)
  ```bash
  python test_face_recognition.py --test_data "lensless_data/test/ymdct_noisy_npy_10px_per_block" --weights "weights/pretrained_weights.pth" --noise_locs "data/noise_locations/noise_10px_per_block.npy"  --batch_size 64
  ```
- ### Face Verification
  ```bash
   python test_face_verification.py --test_data "lensless_data/test/ymdct_npy" --pairs "data/verification_pairs.txt" --weights "weights/pretrained_weights.pth" --noise_locs "data/noise_locations/noise_10px_per_block.npy"  --batch_size 64
  ```
  [test_face_verification.py](https://github.com/engrchrishenry/lensless_face_recognition/blob/main/test_face_verification.py) outputs a 'results.json' file containing 'true_labels' and 'pred_scores' which could be used for computing TPR, FPR, and AUC.

## Citations

If you use this work, please cite:

#### Journal Version
```bibtex
@ARTICLE{10793399,
  author={Henry, Chris and Salman Asif, M. and Li, Zhu},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Privacy-Preserving Face Recognition and Verification With Lensless Camera}, 
  year={2025},
  volume={7},
  number={3},
  pages={354-367},
  keywords={Face recognition;Discrete cosine transforms;Cameras;Privacy;Image recognition;Training;Noise;Image reconstruction;Lensless camera;FlatCam;face recognition;face verification;visual privacy;discrete cosine transform},
  doi={10.1109/TBIOM.2024.3515144}}
```
#### Conference Version
```bibtex
@INPROCEEDINGS{10096627,
  author={Henry, Chris and Asif, M. Salman and Li, Zhu},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Privacy Preserving Face Recognition with Lensless Camera}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Privacy;Face recognition;Speech recognition;Cameras;Time measurement;Discrete cosine transforms;Lensless camera;FlatCam;face recognition;visual privacy;DCT},
  doi={10.1109/ICASSP49357.2023.10096627}}
```

## Acknowledgements

This research was supported in part by the **National Science Foundation (NSF)** under Award **2148382** and Award **2133084**.

Parts of this codebase are adapted from or inspired by the FlatCam implementation available at:

https://github.com/tanjasper/flatcam

We thank the authors and contributors for making their work publicly available.

## Contact
In case of questions, feel free to contact at chffn@umsystem.edu or engr.chrishenry@gmail.com
