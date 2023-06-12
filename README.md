# AI Companion

AI Companion that listens to you and talks to you.

## How to Use

You can run it on your laptop or try it out via this [website](https://huggingface.co/spaces/Icar/AICompanion)!

### Running on your laptop:

- Clone the repo: `git clone https://github.com/CynapticsAI/AICompanion.git`
- Change Directory: `cd AIcompanion`
- Install Requirements: `pip install -r requirements.txt`
- Run!: `python console.py`
- Optionally, You can run the interface version: `python interactive.py`

### Using the Web Version:

- Visit [website](https://huggingface.co/spaces/Icar/AICompanion)
- Click on "Record from microphone" and click the "Submit" button
- Additionally, you can record a fact about the AI, for example, "I am 40 years old", "I have been a teacher for more than 10 years", and click the add fact button to add the fact about the AI.
- You can also use the text box provided to submit inputs. Type out your message, and press Enter on your keyboard/"Submit" button to submit text.
- You can also add facts via the text box, simply input the fact in the text box and press add fact.
- The output will be displayed in the chat interface and audio will be available in the audio block above the buttons.

<img width="1280" alt="interactive" src="https://github.com/CynapticsAI/AICompanion/assets/95569637/268412b5-1d13-43e1-a2be-f0af0cab5c2c">


## Components

* **OpenAI Whisper-Tiny** : Converts Speech to Text. Implemented Using Huggingface ASR Pipeline
* **PersonaGPT** : Responds to User Input. Implemented Using Huggingface Conversational Pipeline.
* **gTTS** : Converts Text to Speech. Implemented Using gTTS API.

## Contributers

* [Yatharth Gupta](https://github.com/Warlord-K)
* [Vishnu V Jaddipal](https://github.com/Gothos)
* [Samip Shah](https://github.com/snarkyidiot)
* [Yash Vashishtha](https://github.com/Yashiiti)
* [Shivankar Sharma](https://github.com/Shivankar007)
