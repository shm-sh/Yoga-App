import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from yoga_pose_library import integrate_pose_library
from ai_recommendation import get_ai_recommendation, show_recommendation_ui
from yoga_nidra import show_yoga_nidra_interface

# ====================== APP CONFIGURATION ======================
st.set_page_config(
    page_title="YOGGI - Yoga Companion",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== POSE DETECTION SETUP ======================
MODEL_PATH = "C:\\Users\\shaur\\OneDrive\\Desktop\\Abhishekpandey1909-SwasthaVerse-7acaa7a\\Abhishekpandey1909-SwasthaVerse-7acaa7a\\yoga_pose_neural_network_model.h5"
LABELS = ["downdog", "goddess", "plank", "tree", "warrior2"]

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    model = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ====================== CORE FUNCTIONS (UNCHANGED) ======================def extract_landmarks(results):
"""Extract pose landmarks as a flattened array"""
if results.pose_landmarks:
    return np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten()
return np.zeros(132)  # 33 landmarks * 4 values


def get_pose_feedback(predicted_pose, target_pose, confidence):
    """Generate real-time feedback based on pose detection"""
    if confidence < 70:
        return "No pose detected", (255, 0, 0)  # Red
    elif predicted_pose != target_pose:
        return f"Adjust to {target_pose}", (255, 165, 0)  # Orange
    elif confidence < 85:
        return "Good form!", (255, 255, 0)  # Yellow
    else:
        return "Perfect!", (0, 255, 0)  # Green


def process_frame(frame, target_pose):
    """Process webcam frame for pose detection and feedback"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    annotated_image = image.copy()

    if results.pose_landmarks:
        # Draw pose landmarks with custom styling
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            )
        )

        # Get pose prediction
        landmarks = extract_landmarks(results)
        if landmarks.sum() != 0 and model:
            landmarks = landmarks.reshape(1, -1)
            predictions = model.predict(landmarks, verbose=0)
            pred_index = np.argmax(predictions)
            predicted_pose = LABELS[pred_index]
            confidence = predictions[0][pred_index] * 100

            # Generate feedback
            feedback, color = get_pose_feedback(predicted_pose, target_pose, confidence)

            # Add text overlay
            y_position = 50
            for text in [
                f"Target: {target_pose}",
                f"Detected: {predicted_pose} ({confidence:.1f}%)",
                feedback
            ]:
                cv2.putText(annotated_image, text, (20, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_position += 40
    else:
        cv2.putText(annotated_image, "No person detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return annotated_image


# ====================== STREAMLIT UI ======================
def home_tab():
    """Main interface for pose correction"""
    st.header("Real-Time Yoga Pose Correction")

    # Mobile-responsive columns
    col1, col2 = st.columns(2)
    with col1:
        pose_category = st.selectbox(
            "Select Category",
            ["Beginner", "Intermediate", "Advanced", "Surya Namaskar"]
        )
    with col2:
        pose_options = {
            "Beginner": ["Downdog", "Tree", "Child's Pose"],
            "Intermediate": ["Warrior II", "Plank", "Cobra"],
            "Advanced": ["Headstand", "Crow Pose", "Wheel"],
            "Surya Namaskar": ["Full Sequence"]
        }
        target_pose = st.selectbox("Select Pose", pose_options[pose_category])

        # webcam integration
        if st.button("Start Webcam Session"):
            cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break

                processed_frame = process_frame(frame, target_pose.lower())
                frame_placeholder.image(processed_frame, channels="RGB")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


# ====================== UPDATED CSS ======================
def inject_custom_css():
    """Mobile-optimized CSS styles"""
    st.markdown("""
    <style>
    /* Mobile-first base styles */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .header-card {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }

    .stButton>button {
        width: 100%;
        padding: 0.6rem 1rem;
        font-size: 14px;
    }

    /* Desktop styles */
    @media (min-width: 768px) {
        .header-card {
            padding: 2rem;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: auto;
            padding: 0.8rem 1.5rem;
        }
    }

    /* Mobile-specific adjustments */
    @media (max-width: 767px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
        .stSelectbox, .stTextInput {
            width: 100% !important;
        }
        .webcam-container {
            margin: 1rem 0;
            border-radius: 10px;
        }
        .stMarkdown h1 {
            font-size: 1.5rem !important;
        }
        .stButton>button {
            font-size: 16px !important;
        }
    }

    /* Common styles */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application structure"""
    # Mobile viewport meta tag
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """, unsafe_allow_html=True)

    st.title("🧘 YogaCompanion - AI Yoga Assistant")
    st.caption("B.Tech Major Project under the guidance of Prof. Shivesh Sharma")

    inject_custom_css()

    # Navigation sidebar
    with st.sidebar:
        # st.image("logo.png", width=150)
        selected = option_menu(
            "Main Menu",
            ["Home", "AI Recommendations", "Pose Library", "Yoga Nidra"],
            icons=["house", "robot", "book", "moon"],
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "16px"}
            }
        )

    # Page routing
    if selected == "Home":
        home_tab()
    elif selected == "Pose Library":
        # st.write("Pose library implementation goes here")
        integrate_pose_library()
    elif selected == "Yoga Nidra":
        show_yoga_nidra_interface()
    elif selected == "AI Recommendations":
        st.header("✨ AI-Powered Recommendations")
        user_input = show_recommendation_ui()
        if user_input:
            with st.spinner("Generating personalized recommendations..."):
                recommendations = get_ai_recommendation(user_input)
                st.markdown(recommendations)
                st.divider()
                st.caption("ℹ️ These recommendations are generated by Google Gemini AI")


if __name__ == "__main__":
    main()