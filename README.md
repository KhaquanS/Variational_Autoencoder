# Variational_Autoencoder
Implementation of a basic VAE from scratch in pytorch

Use train.py to train the VAE. For example (feel free to experiment with other args too):
!python3 train.py --save_model True

Use infer.py to run an inference on a trained VAE:
!python3 infer.py --model_path "models/vae.py" --num_samples 15 --save_samples True

