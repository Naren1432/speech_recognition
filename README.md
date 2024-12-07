# SPeaker_IdentificationShapeReport for Speaker Identification 

      Using MFCC Features and LSTM 

 

Speaker Identification Using LSTM Networks: 

https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset 

 

Speaker identification involves recognizing a speaker’s identity from their voice, critical for applications like security, voice assistants, and 

automated transcription. 

This project leverages Long Short-Term Memory (LSTM) networks to classify speakers based on their vocal characteristics. It uses Mel-Frequency Cepstral Coefficients (MFCCs)-a robust feature set for capturing speech patterns—as 

input. 

By focusing on accurate speaker classification, this approach has practical 

applications in biometric authentication, personalized systems, and automated speech analytics, addressing the growing demand for reliable voice-based 

technologies. 

 

 

Dataset: 

 

The dataset comprises audio files from five prominent speakers: 

Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret 

Thatcher, and Nelson Mandela. Each file is in WAV format with a sampling rate of 16 kHz. 

 

Key Statistics: 

 

Speakers: 5 

Files per Speaker: 120 

Total Files: 600 

File Duration: ~1 second each 

 

 

ShapeThe data was manually collected, ensuring clear speech samples for each speaker. Below is a summary table: 

 

Text Box 

 

| Benjamin Netanyahu 

| 120 

| 120 

| 

| Jens Stoltenberg 

| 120 

| 120 

| 

| Julia Gillard 

| 120 

| 120 

| 

| Margaret Thatcher 

| 120 

| 120 

| 

| Nelson Mandela 

| 120 

| 120 

| 
Experimental Setup: 

The methodology for speaker classification involves the following steps: 

Feature Extraction 

MFCC Features: Extracted 13 coefficients per audio file to represent key spectral characteristics of speech. 

Normalization: Standardized the features using Standard Scaler for consistency across inputs. 

Model Design 

Input: Sequential MFCC features. 

LSTM Layer: 128 units to capture temporal patterns in the audio. 

Dense Layer: 64 units with ReLU activation for feature refinement. 

Output Layer: 5 units with Softmax activation to classify speakers. 

Training Configuration 

Optimizer: Adam with default learning rate (0.001). 

Loss Function: Sparse categorical cross entropy for multi-class classification. 

Batch Size: 32. 

Epochs: 20, with early stopping (patience of 2 epochs) to avoid overfitting. 

This streamlined approach leverages MFCC features and LSTM networks to effectively model and classify speaker identities. 

 

 

Results: 

The results of speaker classification are summarized as follows: 

 

| 

Metric 

| 

Value 

| 

| 

 

| 

 

| 

| Training Accuracy	|	~96%	| 

| Validation Accuracy |	~93%	| 

| Test Accuracy	| ~91%	| 

| Weighted F1 Score	| ~0.90	| 

Confusion Matrix: 

|	True\Predicted	| Benjamin Netanyahu | Jens Stoltenberg | Julia Gillard | Margaret Thatcher | Nelson Mandela | 

|	|	|	|	|	| 

ShapeText Box| 

 | *Julia Gillard*	| 1	| 1	| 23	| 0	| 1 

| 

| *Margaret Thatcher* | 1	| 0	| 1	| 22	| 1 

| 

| *Nelson Mandela*	| 0	| 0	| 0	| 2	| 23 

| 

From the results, the LSTM model demonstrated strong performance in 

identifying speakers. However, minor confusions occurred between similar- sounding voices, likely due to overlapping vocal features. 

 

 

Conclusion: 

This project demonstrated the efficacy of LSTM networks for speaker 

identification using MFCC features, achieving an impressive accuracy of ~91%. The results highlight the model's potential for practical applications in voice- based technologies. 

Future enhancements could focus on: 

Expanding the dataset with more diverse speakers. 

Exploring alternative features like spectrograms or wavelet transforms. 

Fine-tuning hyperparameters for optimized performance. 

Investigating ensemble methods to further boost accuracy. 

These advancements can pave the way for more robust and versatile speaker identification systems. 

 

 

Instructions for Code Execution (Readme) 

Clone the repository and navigate to the Codes folder: 

git clone <repository-link> 

cd Codes 

Install required dependencies: 

pip install -r requirements.txt 

Train the model (includes feature extraction): 

python train_model.py 

Run inference on an audio file: 

python infer.py --file <path-to-audio-file> 

 

Link to Trained Models and Intermediate Files# speech_recognition
