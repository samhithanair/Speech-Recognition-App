import streamlit as st
import speech_recognition as sr
import nltk
from nltk.corpus import cmudict
from difflib import SequenceMatcher
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import os
# Download CMU Pronouncing Dictionary
nltk.download('cmudict')
d = cmudict.dict()

# Gemini API configuration
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def convert_to_phonetics(word):
    phonetics = d.get(word.lower(), None)
    if phonetics:
        return ' '.join([' '.join(p) for p in phonetics])
    return None

def compare_phonetics(original_phonetics, spoken_phonetics):
    similarity = SequenceMatcher(None, original_phonetics, spoken_phonetics).ratio()
    return similarity

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please say the word...")
        audio = recognizer.listen(source)
        try:
            spoken_word = recognizer.recognize_google(audio)
            st.write(f"Recognized word: {spoken_word}")
            return spoken_word
        except sr.UnknownValueError:
            st.write("Could not understand the audio.")
            return None
        except sr.RequestError:
            st.write("Could not request results from the Google Speech Recognition service.")
            return None

def generate_word(difficulty="easy"):
    prompt = f"""Generate a single {difficulty} English word for pronunciation practice. The word should be able to be said by 8-10 year olds.
    The word should be appropriate for the difficulty level:
    - Easy: common, short words with simple pronunciation
    - Medium: medium length words with more complex phonetics
    - Hard: challenging words with difficult letter combinations.
    
    Return only the word, nothing else."""
    
    try:
        response = model.generate_content(prompt)
        word = response.text.strip().lower()
        # Remove any punctuation or extra whitespace
        word = ''.join(c for c in word if c.isalnum())
        return word
    except Exception as e:
        st.error(f"Error generating word: {str(e)}")
        return "hello"  # fallback word

def test_pronunciation(word):
    original_phonetics = convert_to_phonetics(word)
    st.write(f"Original word phonetics: {original_phonetics}")
    
    spoken_word = recognize_speech()
    if spoken_word:
        spoken_phonetics = convert_to_phonetics(spoken_word)
        st.write(f"Spoken word phonetics: {spoken_phonetics}")
        
        if original_phonetics and spoken_phonetics:
            similarity = compare_phonetics(original_phonetics, spoken_phonetics)
            st.write(f"Similarity score: {similarity}")
            
            if similarity > 0.8:
                st.success("Good pronunciation!")
                return True
            else:
                st.warning("Please try again.")
                # Get pronunciation tips from Gemini
                tips_prompt = f"""Provide a short, specific tip for pronouncing the word '{word}' correctly. 
                Focus on the most challenging part of the pronunciation.
                Keep the response under 50 words."""
                try:
                    tips_response = model.generate_content(tips_prompt)
                    st.info(f"Pronunciation tip: {tips_response.text}")
                except Exception as e:
                    st.error(f"Error getting pronunciation tips: {str(e)}")
                return False
        else:
            st.warning("Phonetics could not be generated for one or both words.")
    else:
        st.warning("No word recognized.")
    return False

def main():
    st.title("Pronunciation Practice")
    st.write("Test your pronunciation by speaking into the microphone.")
    
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "easy"
    
    # Add difficulty selection
    difficulty_options = ["easy", "medium", "hard"]
    selected_difficulty = st.selectbox(
        "Select difficulty level:",
        difficulty_options,
        index=difficulty_options.index(st.session_state.difficulty)
    )
    st.session_state.difficulty = selected_difficulty
    
    if st.button("Generate Word"):
        word = generate_word(st.session_state.difficulty)
        st.session_state.word = word
        st.write(f"Your word is: {word}")
        
        # Get word usage example from Gemini
        example_prompt = f"Write a short, simple sentence using the word '{word}' in context."
        try:
            example_response = model.generate_content(example_prompt)
            st.write(f"Example: {example_response.text}")
        except Exception as e:
            st.error(f"Error getting example sentence: {str(e)}")
    
    if "word" in st.session_state:
        if st.button("Test Pronunciation"):
            success = test_pronunciation(st.session_state.word)
            if success:
                # Increase difficulty if pronunciation was good
                current_index = difficulty_options.index(st.session_state.difficulty)
                if current_index < len(difficulty_options) - 1:
                    st.session_state.difficulty = difficulty_options[current_index + 1]
                    st.info(f"Great job! Difficulty increased to {st.session_state.difficulty}")
            else:
                # Get similar word suggestion for practice
                similar_prompt = f"""Suggest a single word that has a similar pronunciation pattern to '{st.session_state.word}' 
                that might help with practice. Return only the word."""
                try:
                    similar_response = model.generate_content(similar_prompt)
                    st.write("Try practicing with this similar word:")
                    st.write(similar_response.text.strip())
                except Exception as e:
                    st.error(f"Error getting similar word suggestion: {str(e)}")

if __name__ == "__main__":
    main()
