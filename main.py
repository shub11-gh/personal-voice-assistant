import speech_recognition
import pyttsx3 as tts
import sys
import nltk
import datetime
import os
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')

# ── TTS & Recognizer setup ──────────────────────────────────────────
recognizer = speech_recognition.Recognizer()
speaker = tts.init()
speaker.setProperty('rate', 150)

# ── Data ────────────────────────────────────────────────────────────
todo_list = ['Go Shopping', 'Clean Room']

periodic_table = {
    'hydrogen': 'Hydrogen is the first element with symbol H and atomic number 1.',
    'helium':   'Helium is a noble gas with symbol He and atomic number 2.',
    'oxygen':   'Oxygen is essential for respiration. Symbol O, atomic number 8.',
    'carbon':   'Carbon is the basis of life. Symbol C, atomic number 6.',
    'iron':     'Iron has symbol Fe and atomic number 26. Used to make steel.'
}

motivation_quotes = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Success is not the key to happiness. Happiness is the key to success.",
    "Don't watch the clock; do what it does. Keep going.",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "The future depends on what you do today. - Mahatma Gandhi"
]

concepts = {
    'photosynthesis': 'Photosynthesis is the process by which green plants use sunlight to make food.',
    'gravity':        'Gravity is a force that attracts two bodies toward each other.',
    'evaporation':    'Evaporation is the process by which water changes from liquid to gas.',
    'atom':           'An atom is the smallest unit of matter that forms a chemical element.'
}

# ── Lightweight Intent Classifier (replaces neuralintents) ──────────
class LightAssistant:
    def __init__(self, intents_file, method_mappings={}):
        self.method_mappings = method_mappings

        with open(intents_file) as f:
            self.intents = json.load(f)['intents']

        self.patterns = []
        self.tags = []

        for intent in self.intents:
            for pattern in intent['patterns']:
                self.patterns.append(pattern.lower())
                self.tags.append(intent['tag'])

        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.patterns)

    def process_input(self, message):
        message_vec = self.vectorizer.transform([message.lower()])
        similarities = cosine_similarity(message_vec, self.X)
        best_idx = np.argmax(similarities)
        best_score = similarities[0][best_idx]

        if best_score < 0.1:
            return "I'm sorry, I didn't understand that."

        matched_tag = self.tags[best_idx]

        if matched_tag in self.method_mappings:
            self.method_mappings[matched_tag]()
            return None

        for intent in self.intents:
            if intent['tag'] == matched_tag:
                responses = intent['responses']
                return random.choice(responses) if responses else None

        return None

# ── Assistant Functions ─────────────────────────────────────────────
def create_note():
    global recognizer
    speaker.say("What do you want to write on your note?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.3)
                audio = recognizer.listen(mic)
                note = recognizer.recognize_google(audio).lower()

                speaker.say("Choose a file name!")
                speaker.runAndWait()
                recognizer.adjust_for_ambient_noise(mic, duration=0.3)
                audio = recognizer.listen(mic)
                filename = recognizer.recognize_google(audio).lower()

            with open(filename + ".txt", 'w') as f:
                f.write(note)
            done = True
            speaker.say(f"I successfully created the note {filename}")
            speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you! Please try again")
            speaker.runAndWait()

def add_todo():
    global recognizer
    speaker.say("What do you want to add?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.3)
                audio = recognizer.listen(mic)
                item = recognizer.recognize_google(audio).lower()
            todo_list.append(item)
            done = True
            speaker.say(f"I added {item} to the to do list!")
            speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you! Please try again")
            speaker.runAndWait()

def show_todos():
    speaker.say("The items on your to do list are: " + ' '.join(todo_list))
    speaker.runAndWait()

def hello():
    speaker.say("Hello! What can I do for you?")
    speaker.runAndWait()

def exit_assistant():
    speaker.say("Thank you! Bye")
    speaker.runAndWait()
    sys.exit(0)

def periodic_fact():
    speaker.say("Which element do you want to know about?")
    speaker.runAndWait()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.3)
        audio = recognizer.listen(mic)
        element = recognizer.recognize_google(audio).lower()
    fact = periodic_table.get(element, f"Sorry, I don't have info about {element}.")
    speaker.say(fact)
    speaker.runAndWait()

def motivate():
    speaker.say(random.choice(motivation_quotes))
    speaker.runAndWait()

def explain_concept():
    speaker.say("Which concept do you want me to explain?")
    speaker.runAndWait()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.3)
        audio = recognizer.listen(mic)
        concept = recognizer.recognize_google(audio).lower()
    speaker.say(concepts.get(concept, f"Sorry, I don't have an explanation for {concept}."))
    speaker.runAndWait()

def tell_date():
    speaker.say(datetime.datetime.now().strftime("Today is %A, %B %d, %Y."))
    speaker.runAndWait()

def tell_time():
    speaker.say(datetime.datetime.now().strftime("The time is %I:%M %p."))
    speaker.runAndWait()

def delete_todo():
    speaker.say("Which to-do do you want to delete?")
    speaker.runAndWait()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.3)
        audio = recognizer.listen(mic)
        item = recognizer.recognize_google(audio).lower()
    if item in todo_list:
        todo_list.remove(item)
        speaker.say(f"Removed {item} from your to-do list.")
    else:
        speaker.say(f"{item} is not in your to-do list.")
    speaker.runAndWait()

def list_notes():
    notes = [f for f in os.listdir('.') if f.endswith('.txt')]
    speaker.say("Your notes are: " + ' '.join(notes) if notes else "You have no notes.")
    speaker.runAndWait()

def read_note():
    speaker.say("Which note should I read? Say the file name without dot txt.")
    speaker.runAndWait()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.3)
        audio = recognizer.listen(mic)
        filename = recognizer.recognize_google(audio).lower() + ".txt"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            speaker.say(f.read())
    else:
        speaker.say(f"Note {filename} does not exist.")
    speaker.runAndWait()

# ── Start Assistant ─────────────────────────────────────────────────
assistant = LightAssistant('intents.json', method_mappings={
    "greeting":       hello,
    "create_note":    create_note,
    "add_todo":       add_todo,
    "show_todos":     show_todos,
    "exit":           exit_assistant,
    "periodic_fact":  periodic_fact,
    "motivate":       motivate,
    "explain_concept": explain_concept,
    "tell_date":      tell_date,
    "tell_time":      tell_time,
    "delete_todo":    delete_todo,
    "list_notes":     list_notes,
    "read_note":      read_note
})

print("Assistant is ready! Start speaking...")

# ── Main Loop ───────────────────────────────────────────────────────
while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.3)
            audio = recognizer.listen(mic)
            message = recognizer.recognize_google(audio).lower()
            print(f"You said: {message}")

        response = assistant.process_input(message)
        if response:
            print(f"Assistant: {response}")
            speaker.say(response)
            speaker.runAndWait()

    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()