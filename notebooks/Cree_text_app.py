# cree_app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from cree_learning_model import CreeLearningModel

st.set_page_config(page_title="Cree Language Learning App", layout="wide")
st.title("🌿 Cree Language Learning App")
# Load the model
@st.cache_resource
def load_model():
    model = CreeLearningModel()
    model.load_model("../models/cree_learning_model.pkl")
    return model

model = load_model()



# Sidebar
mode = st.sidebar.selectbox("Choose Mode", [
    "Translate", "Exercise", "Dataset Explorer"])

# --- Translate Mode ---
if mode == "Translate":
    st.header("🔁 Translation")
    direction = st.radio("Translation Direction", ["Cree → English", "English → Cree"])
    input_word = st.text_input("Enter word:")
    if st.button("Translate"):
        if direction == "Cree → English":
            translations = model.find_translations(input_word)
        else:
            translations = model.find_cree_words(input_word)
        st.write("### Translations:", translations or "No match found.")

# --- Exercise Mode ---
elif mode == "Exercise":
    st.header("🧪 Take a Cree Translation Test")

    difficulty = st.selectbox("Choose difficulty:", ["mixed", "easy", "hard"])

    # --- Reset on restart or first time ---
    if "test_initialized" not in st.session_state or st.session_state.get("difficulty_mode") != difficulty:
        st.session_state.test_questions = model.create_learning_exercises(difficulty=difficulty)[:10]
        st.session_state.current_q = 0
        st.session_state.submitted_answers = [None] * 10
        st.session_state.correct_flags = [False] * 10
        st.session_state.feedbacks = [""] * 10
        st.session_state.finished = False
        st.session_state.difficulty_mode = difficulty
        st.session_state.test_initialized = True

    q_idx = st.session_state.current_q
    question = st.session_state.test_questions[q_idx]

    # --- If test is not finished ---
    if not st.session_state.finished:
        st.markdown(f"### Question {q_idx + 1} of 10")
        st.markdown(f"**Translate this Cree word:** `{question['cree_word']}`")

        selected = st.radio(
            "Choose your answer:",
            options=question["choices"],
            key=f"choice_q{q_idx}"
        )

        already_submitted = st.session_state.submitted_answers[q_idx] is not None
        if already_submitted:
            st.info(st.session_state.feedbacks[q_idx])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Submit Answer"):
                if not already_submitted:
                    st.session_state.submitted_answers[q_idx] = selected
                    if selected in question["correct_answers"]:
                        st.session_state.correct_flags[q_idx] = True
                        st.session_state.feedbacks[q_idx] = "✅ Correct!"
                    else:
                        correct = ", ".join(question["correct_answers"])
                        st.session_state.feedbacks[q_idx] = f"❌ Incorrect. Correct answer: **{correct}**"
                    st.rerun()

        with col2:
            if already_submitted and st.button("➡️ Next Question"):
                if q_idx < 9:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.warning("This is the last question.")

        st.markdown("---")
        if st.button("🏁 Evaluate Test"):
            st.session_state.finished = True
            st.rerun()

    # --- When test is finished ---
    else:
        total_correct = sum(st.session_state.correct_flags)
        attempted = sum(ans is not None for ans in st.session_state.submitted_answers)

        st.success(f"✅ You answered {total_correct} out of 10 questions correctly.")
        st.write(f"🧮 You attempted {attempted} questions.")
        st.write(f"📊 Your score: **{total_correct} / 10**")
        st.markdown("---")

        for i, q in enumerate(st.session_state.test_questions):
            user_answer = st.session_state.submitted_answers[i]
            correct = ", ".join(q["correct_answers"])
            st.markdown(f"**Q{i+1}.** `{q['cree_word']}` → Your answer: `{user_answer or 'Not answered'}`")
            if user_answer is None:
                st.markdown("🟡 Not attempted")
            elif user_answer in q["correct_answers"]:
                st.markdown("✅ Correct!")
            else:
                st.markdown(f"❌ Incorrect. Correct answer: **{correct}**")
            st.markdown("—")

        if st.button("🔁 Restart Test"):
            for key in ["test_initialized", "current_q", "submitted_answers", "correct_flags", "feedbacks", "finished", "difficulty_mode"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# --- Dataset Explorer ---
elif mode == "Dataset Explorer":
    st.header("📚 Dataset Overview")
    cree_words = list(model.cree_to_english.keys())
    word_limit = st.slider("How many Cree words to show:", 5, 1000, 10)
    data = [(w, ", ".join(model.cree_to_english[w])) for w in cree_words[:word_limit]]
    df = pd.DataFrame(data, columns=["Cree Word", "English Meanings"])
    st.dataframe(df)

    st.write("---")
    st.write(f"Total Cree words: {len(model.cree_to_english)}")
    st.write(f"Cree words with multiple meanings: {sum(1 for v in model.cree_to_english.values() if len(v) > 1)}")
    st.write(f"Average meanings per Cree word: {np.mean([len(v) for v in model.cree_to_english.values()]):.2f}")
