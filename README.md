## Self-Supervised Pre-Training for Covid-19 Heart Pathology Identification




Steps:
1. Train baseline target model using covid small train and validation (labelled)
2. Self-supervised auxiliary pre-training by (choose one):
    - noise/denoise (how much noise to add?)
    - inpainting (patch completion)
    - ...?
3. Use weights from pre-training to re-train target model 
    - compare training speed, absolute loss (performance)
    - visual (subjective) comprarison
    - 3d visulations in itk / blender / ....?


General Notes:
- work on local gpu (stevan)
    - implement both for CUDA and non CUDA 
- download dataset (covid small) to local env