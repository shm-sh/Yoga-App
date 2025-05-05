# ai_recommendation.py
import google.generativeai as genai
import streamlit as st

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


def get_ai_recommendation(user_input):
    """Get yoga recommendations using Google Gemini"""
    prompt = f"""
    As a professional yoga instructor, recommend 3 personalized poses for:
    - Age: {user_input['age']}
    - Fitness Level: {user_input['fitness_level']}
    - Health Conditions: {', '.join(user_input['health_conditions'])}
    - Goals: {user_input['goals']}

    Format each recommendation as:
    **Pose Name**: Benefits (Difficulty: Beginner/Intermediate/Advanced)

    Include modifications if relevant for health conditions.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return get_fallback_recommendations(user_input)


def get_fallback_recommendations(user_input):
    """Local backup recommendations"""
    poses = {
        "Beginner": [
            "**Mountain Pose**: Improves posture (Beginner)",
            "**Child's Pose**: Relieves stress (Beginner)"
        ],
        "Intermediate": [
            "**Warrior II**: Strengthens legs (Intermediate)",
            "**Tree Pose**: Improves balance (Intermediate)"
        ],
        "Advanced": [
            "**Headstand**: Boosts circulation (Advanced)",
            "**Crow Pose**: Builds arm strength (Advanced)"
        ]
    }
    return "\n".join(poses.get(user_input["fitness_level"], []))


def show_recommendation_ui():
    """Streamlit form for user input"""
    with st.form("user_profile"):
        st.subheader("🧘 Personal Profile")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 80, 30)
            fitness_level = st.selectbox(
                "Fitness Level",
                ["Beginner", "Intermediate", "Advanced"]
            )
        with col2:
            health_conditions = st.multiselect(
                "Health Considerations",
                ["Back Pain", "Hypertension", "Pregnancy", "Joint Issues", "None"]
            )
            goals = st.selectbox(
                "Primary Goal",
                ["Stress Relief", "Flexibility", "Strength", "Rehabilitation"]
            )

        if st.form_submit_button("Get AI Recommendations"):
            return {
                "age": age,
                "fitness_level": fitness_level,
                "health_conditions": health_conditions,
                "goals": goals
            }
    return None