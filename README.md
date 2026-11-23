# MediChat Chatbot: Training and Design

My part of the **MediChat AI Consultant** capstone (Spring 2025, Manhattan University): the chatbot, built end to end, from data filtering and model training to the hybrid chat system that was handed to the frontend team.

**Full project:** [gitlab.com/petra184/medi-chat](https://gitlab.com/petra184/medi-chat). That repository holds the complete application and the work of the whole team, so please look there for the project as a whole and for what each member built.

## The project

MediChat AI Consultant is a web application that pairs an interactive 3D model of the human body with an AI chatbot. Users select body parts, describe their symptoms, and receive possible conditions along with treatment or first-aid suggestions; the app also stores their chat and medical history. The team built the frontend with Next.js and Three.js and the backend with FastAPI, and used Supabase for data. This repository contains only the chatbot, which I designed, trained and built on my own, and supplied to the frontend team for integration.

## My contribution

I am the sole author of the chatbot process in this repository:

- **Data:** selected and filtered the datasets for muscle-related content and prepared them for training.
- **Training:** fine-tuned four candidate models locally, chose RoBERTa, and extended its training on the university's Kakos cluster.
- **Chatbot:** designed and built the hybrid chatbot, a fine-tuned model with a knowledge-base guide and an LLM that polishes the output, across several iterations.
- **Delivery:** shipped working API versions early and kept the interface stable so the frontend team could build in parallel while the model improved.

## Repository structure

```
├── 01_data_and_preprocessing/    Dataset credits and the filtering/preparation script
├── 02_training_and_selection/    Local training of four models, the selection, and cluster training
│   ├── local_training.py
│   └── cluster_training_artemis.py
└── 03_chatbot_design/            The chatbot in four stages, each with its own README
    ├── 01_api_baseline/
    ├── 02_hybrid_polish_experiments/
    ├── 03_nltk_symptom_matching/
    └── 04_final_mistral_api/
```

Read in order, the folders follow the process: data, then training and selection, then the chatbot.

## Results in brief

On a small test over 100 muscle-related entries, run in both directions (disease to symptoms, and symptoms to disease), RoBERTa was the strongest candidate locally with 31 hits. After extended training on the cluster it reached 56. This is a project-level sanity check, not a formal benchmark. Details are in [02_training_and_selection](02_training_and_selection/).

## Notes

- **Scope:** an engineering project, built in one semester to deliver an end-to-end product. I am continuing this work in my own research.
- **Not a medical device:** the chatbot's output is not clinically validated and must not be used for medical decisions.
- **Not included:** datasets (see the credits and licensing notes in `01_data_and_preprocessing`), the trained model weights, and all API keys. Keys in the code are replaced with environment variables or placeholders.
- **Reconstructed code:** some training code was consolidated or reconstructed from staged runs. Each file's header says exactly what, and the training README summarizes it.
- **The capstone report:** the report submitted for the course was a condensed version and is not included. Where it differs from this repository, this repository is the accurate account.

## Credits

Thank you to my capstone teammates, **Chelsy Jimenez**, **Joseph Beauchamp** and **Petra Juric**, for the whole project. Thank you to the creators of the datasets and models listed in [01_data_and_preprocessing](01_data_and_preprocessing/).

*Bhavya Mishra*
