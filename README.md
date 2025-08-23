# Fine-Tuning-SmolVLA-450M-for-Industrial-Robotic-Manipulation


smolvla-industrial-robotics/
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── data/
│   ├── synthetic_data_generator.py
│   └── requirements_synthetic.txt
├── training/
│   ├── train.py
│   ├── dataset.py
│   └── requirements_train.txt
├── inference/
│   ├── inference.py
│   ├── safety_wrapper.py
│   └── requirements_inference.txt
├── simulation/
│   ├── pybullet_sim.py
│   └── requirements_sim.txt
├── utils/
│   ├── image_utils.py
│   └── action_utils.py
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
