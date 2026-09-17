# NMR-UNet-Denoiser
A TensorFlow/Keras 1D U-Net to denoise real NMR signals (FIDs) and spectra from severe Gaussian noise, preserving physical parameters (T2*, frequency, power). The model achieves median peak signal to noise ratios (PSNR) up to 45 dB, structural similarity indices (SSIM) up to 0.96, and an SNR gain up to 11 dB across an independent test set.

This repository provides the computational pipeline for the article "Robust U-Net Reconstruction of NMR Signals and Spectra in Noisy and Inhomogeneous Magnetic Fields" by Alfredo Aguilar, Lautaro Piermartini, and Esteban Anoardo. Built all in Python, the framework executes the proposed architecture on experimental data, aiming to remove severe noise artifacts without distorting the underlying spin physics.

================================================================================

TECHNICAL INSTRUCTION: U-NET 1D DENOISING PIPELINE

================================================================================

--------------------------------------------------------------------------------
1. GENERAL DESCRIPTION

This repository consists of a main notebook ready to be uploaded and executed directly on the Google Colab root without creating prior folders. All required libraries are imported at the beginning of the notebook. The execution environment should support GPU or TPU acceleration (available in Colab) due to the computational demands of the data augmentation and training phases, ensuring true computational efficiency and not just execution speed (not mandatory). Main files are:

- NMR-Denoiser.ipynb: The main notebook dedicated to data processing, data augmentation, training, physical evaluation, and saving the results.
- u_net.py: The complementary module containing the 1D U-Net network architecture and necessary auxiliary function. The upload of this file is explicitly guided by the comments in the corresponding notebook cell.

--------------------------------------------------------------------------------
2. FILE STRUCTURE AND LOADING

The notebook cells will explicitly tell you when to upload the following files to the Colab root (/content/). Please follow the instructions and do not upload them beforehand:

- u_net.py: The module containing the 1D U-Net mathematical architecture and the additive Gaussian noise generator.
- FIDs.zip: A compressed file containing the experimental NMR signals. When unzipped by the script, it creates a folder named "FIDs" containing the raw signal files (1.csv, 2.csv, ...). Each CSV file is structured in two columns: time and amplitude.
- FFTs.zip: A compressed file containing the frequency spectra of the FID signals. When unzipped by the script, it creates a folder named "FFTs" containing the spectrum files (1_FFT.csv, 2_FFT.csv, ...). Each CSV file is structured in two columns: frequency and amplitude.
- times.csv: A plain text file containing the temporal vector corresponding to the acquisition of the FIDs. Since the time axis is identical for all signals, this file is loaded only once. Consequently, only the amplitudes (the second column) are extracted from the individual files within the FIDs folder and associated with this common time vector. This approach avoids loading repeated temporal data, preventing unnecessary slowdowns during model training.

--------------------------------------------------------------------------------
3. DATA ALIGNMENT AND PARTITIONING

- Signal Cropping: The notebook asks for a minimum (t_min) and maximum (t_max) time in seconds (0 - 0.005 s max.). All signals are then automatically cropped to this specific temporal window.
- Time Mapping: The length of the amplitude vectors extracted from the individual CSV files inside the FIDs folder must match the length of the temporal vector provided in times.csv. This alignment is handled automatically by the script; the user does not need to manually adjust or format the data.
- Data Splitting: The script will request an integer (num_train_test) to divide the dataset. All loaded signals are first randomly shuffled. The dataset is then split by allocating the specified number of signals to the training set, leaving the remainder for the test set. Later in the pipeline, after data augmentation, the expanded training block is shuffled again and split (80/20) to create an internal validation set.

--------------------------------------------------------------------------------
4. GOOGLE COLAB EXECUTION INSTRUCTIONS

A. Start a new Google Colab session, look for Hardware Acceleration (GPU/TPU) is enabled for computational efficiency, and upload the main notebook.

B. Execute the cells sequentially. The inline comments and printed text outputs within the notebook will guide you through the process.

C. Module Loading: When prompted by the cell output, upload the "u_net.py" file.

D. Data Loading: A subsequent cell will prompt you to upload the dataset ("FIDs.zip" or "FFTs.zip") and the "times.csv" file. The script handles the decompression automatically.

E. Signal Cropping: The execution will pause and ask via console input for a minimum time value and a maximum time value in seconds. Enter these numbers to define the temporal window. The script will automatically crop all loaded signals to this exact range based on the time vector.

F. Data Splitting: Right after cropping, the script will request an integer via console input. This number defines how many signals from the sequence will be allocated for the training set, leaving the rest for testing.

G. Preprocessing and Augmentation: The notebook automatically drops invalid data (NaNs), applies a 99.5% quantile normalization, fits a StandardScaler on the training set, and expands the dataset by injecting additive Gaussian noise at a predefined list of SNR target levels.

H. Training: The network is compiled and trained automatically using the Adam optimizer, early stopping, and adaptive learning rate reduction based on validation loss.

I. Physical Evaluation: The script runs a complete evaluation on the test set. It calculates general metrics (PSNR, SSIM, RMSE, SNR gain) and computes the physical NMR parameters (T2* relaxation time, peak frequency, FWHM, and main peak spectral power) for both the reference and the denoised signals.

J. Saving and Downloading Results: The model (.h5), scalers (.pkl), hyperparameter configuration (metadata.json), logs, plots, and a batch of test examples in CSV format are saved inside a generated "resultados/" folder. The script then zips this folder into "Resultados.zip" and triggers an automatic download.

K. Environment Cleaning: The final cell executes a cleanup routine that deletes all decompressed datasets, generated folders, and temporary files from the Colab root, leaving the session empty for a new run.

================================================================================

