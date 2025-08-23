# SmolVLA-450M Industrial Robotics Fine-Tuning

This project fine-tunes the SmolVLA-450M model for industrial bin-picking tasks.

## Project Structure

- `configs/`: Configuration files for data generation, model, and training
- `data/`: Synthetic data generation scripts
- `training/`: Model training scripts
- `inference/`: Inference and safety checking
- `simulation/`: PyBullet simulation environment
- `utils/`: Utility functions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/smolvla-industrial-robotics.git
cd smolvla-industrial-robotics






                             smolvla-industrial-robotics/

                             ├── configs/
                                         │   
                                         ├── data_config.yaml
                                         │   
                                         ├── model_config.yaml
                                         │   
                                         └── train_config.yaml
                            ├── data/
                                     │   
                                     ├── synthetic_data_generator.py
                                     │   
                                     └── requirements_synthetic.txt
                                    
                            ├── training/
                                         │   
                                         ├── train.py
                                         │   
                                         ├── dataset.py
                                         │   
                                         └── requirements_train.txt
                                        
                            ├── inference/
                                          │   
                                          ├── inference.py
                                          │   
                                          ├── safety_wrapper.py
                                          │   
                                          └── requirements_inference.txt
                                          
                            ├── simulation/
                                           │   
                                           ├── pybullet_sim.py
                                           │   
                                           └── requirements_sim.txt
                                           
                            ├── utils/
                                      │   
                                      ├── image_utils.py
                                      │   
                                      └── action_utils.py
                                      
                           ├── Dockerfile
                           ├── docker-compose.yml
                           ├── README.md
                           └── requirements.txt
