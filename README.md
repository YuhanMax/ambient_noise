# ambient_noise

## Intro

info

A fun project for generating audio tracks by listening to the environment. 

## Activity

2025-03-25

Testing with generating music similar to **Mother Earth's Plantasia**, which is an electronic album by Mort Garson, released in 1976.

## Reference

https://towardsdatascience.com/writing-a-music-album-with-deep-learning-4ee3bd2e9b05/

https://magenta.tensorflow.org/studio/

plantasia_generator/
│
├── data/
│   ├── raw_midi/           # Store your source MIDI files here
│   ├── processed/          # Processed TFRecord files
│   └── generated/          # Your generated compositions
│
├── models/
│   ├── trained/            # Saved model checkpoints
│   └── bundles/            # Bundled models ready for generation
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Functions for MIDI processing
│   ├── model.py            # Model definition and training code
│   ├── generator.py        # Generation functions
│   └── environment.py      # Environmental sound analysis code
│
├── utils/
│   ├── __init__.py
│   ├── midi_utils.py       # MIDI manipulation utilities
│   └── audio_utils.py      # Audio processing utilities
│
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── data_exploration.ipynb
│   └── generation_examples.ipynb
│
├── scripts/
│   ├── train.py            # Script to train the model
│   ├── generate.py         # Script to generate new compositions
│   └── listen_environment.py # Script for environment listening app
│
├── requirements.txt        # Project dependencies
├── setup.py                # Package installation
└── README.md               # Project documentation