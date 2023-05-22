# AI Companion

AI Companion that listens to you and talks to you.

## How to Use

You can run it on your laptop or try it out via this [website]()!

### Running on your laptop:

- Clone the repo: `git clone https://github.com/CynapticsAI/AICompanion.git`
- Change Directory: `cd AIcompanion`
- Install Requirements: `pip install -r requirements.txt`
- Run!: `python console.py`
- Optionally, You can run the interface version: `python interactive.py`

### Using the Web Version:

- Visit [website]()
- Click on "Record from microphone"
- Click Submit
- The output will be displayed in the chat interface and audio will be available in the audio block above the buttons.
- Additionally, You can record a fact about the AI, for example, "I am 40 years old", "I have been a teacher for more than 10 years", and click the add fact button to add the fact about the AI.

<img width="1280" alt="interactive" src="https://github.com/CynapticsAI/AICompanion/assets/95569637/268412b5-1d13-43e1-a2be-f0af0cab5c2c">


## Components

* **OpenAI Whisper-Tiny** : Converts Speech to Text. Implemented Using Huggingface ASR Pipeline
* **PersonaGPT** : Responds to User Input. Implemented Using Huggingface Conversational Pipeline.
* **gTTS** : Converts Text to Speech. Implemented Using gTTS API.

## Contributers

* [Yatharth Gupta](https://github.com/Warlord-K)
