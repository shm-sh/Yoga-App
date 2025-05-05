# progress.py - Enhanced Yoga Practice Progress Tracking
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from pathlib import Path

# ====================== DATA MANAGEMENT ======================
PROGRESS_FILE = "yoga_progress.json"


def _ensure_data_dir():
    """Ensure data directory exists"""
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)


def load_progress():
    """Load progress data from file or initialize new"""
    _ensure_data_dir()
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                # Convert old date strings to datetime objects
                if data.get("sessions"):
                    for session in data["sessions"]:
                        if isinstance(session["date"], str):
                            session["date"] = datetime.fromisoformat(session["date"])
                return data
    except Exception as e:
        st.error(f"Error loading progress data: {str(e)}")

    return {
        "sessions": [],
        "total_duration": 0,
        "total_calories": 0,
        "pose_accuracy": {},
        "current_streak": 0,
        "last_session_date": None
    }


def save_progress(data):
    """Save progress data to file"""
    try:
        _ensure_data_dir()
        # Convert datetime objects to strings before saving
        data_to_save = data.copy()
        if data_to_save.get("sessions"):
            for session in data_to_save["sessions"]:
                if isinstance(session["date"], datetime):
                    session["date"] = session["date"].isoformat()

        with open(PROGRESS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    except Exception as e:
        st.error(f"Error saving progress data: {str(e)}")


# ====================== CORE FUNCTIONS ======================
def update_progress(session_data):
    """Update progress with new session data"""
    progress_data = load_progress()
    now = datetime.now()

    # Create new session record
    new_session = {
        "date": now,
        "duration": float(session_data["duration"]),
        "calories": float(session_data["calories"]),
        "pose": session_data["pose"],
        "accuracy": float(session_data["accuracy"])
    }

    # Update basic stats
    progress_data["sessions"].append(new_session)
    progress_data["total_duration"] += new_session["duration"]
    progress_data["total_calories"] += new_session["calories"]

    # Update pose-specific accuracy
    pose = new_session["pose"]
    if pose not in progress_data["pose_accuracy"]:
        progress_data["pose_accuracy"][pose] = []
    progress_data["pose_accuracy"][pose].append(new_session["accuracy"])

    # Update streak
    update_streak(progress_data, now.date())

    save_progress(progress_data)
    return progress_data


def update_streak(progress_data, current_date):
    """Calculate and update current streak"""
    if not progress_data["sessions"]:
        progress_data["current_streak"] = 0
        return

    # Get sorted unique dates
    dates = sorted({datetime.fromisoformat(s["date"]).date()
                    if isinstance(s["date"], str)
                    else s["date"].date()
                    for s in progress_data["sessions"]})

    streak = 0
    expected_date = current_date

    # Check backwards from current date
    while expected_date in dates:
        streak += 1
        expected_date -= timedelta(days=1)

    progress_data["current_streak"] = streak


# ====================== UI COMPONENTS ======================
def show_progress_sidebar():
    """Display progress summary in sidebar"""
    try:
        progress_data = load_progress()

        st.sidebar.header("Your Progress")

        # Current streak with conditional formatting
        streak = progress_data.get("current_streak", 0)
        streak_color = "#4CAF50" if streak >= 3 else "#FF9800" if streak > 0 else "#F44336"
        st.sidebar.markdown(
            f"<h3 style='color:{streak_color}'>🔥 {streak} Day Streak</h3>",
            unsafe_allow_html=True
        )

        # Total stats
        st.sidebar.subheader("Lifetime Stats")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total Hours", f"{progress_data['total_duration'] / 60:.1f}")
        col2.metric("Calories", f"{progress_data['total_calories']:.0f} kcal")

        # Recent sessions
        st.sidebar.subheader("Last Session")
        if progress_data["sessions"]:
            last_session = progress_data["sessions"][-1]
            date = (last_session["date"].strftime("%b %d")
                    if isinstance(last_session["date"], datetime)
                    else datetime.fromisoformat(last_session["date"]).strftime("%b %d"))

            st.sidebar.write(f"**{last_session['pose'].title()}**")
            st.sidebar.write(f"📅 {date}")
            st.sidebar.write(f"⏱️ {last_session['duration']} min")
            st.sidebar.write(f"🎯 {last_session['accuracy']:.0f}%")
        else:
            st.sidebar.write("No sessions yet")

    except Exception as e:
        st.sidebar.error(f"Error loading progress: {str(e)}")


def show_progress_main():
    """Main progress tracking interface"""
    try:
        progress_data = load_progress()

        st.header("🧘 Your Yoga Journey")

        # Summary Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", len(progress_data["sessions"]))
        with col2:
            st.metric("Total Hours", f"{progress_data['total_duration'] / 60:.1f}")
        with col3:
            st.metric("Calories Burned", f"{progress_data['total_calories']:.0f}")

        # Weekly Summary Chart
        st.subheader("Weekly Activity")
        if progress_data["sessions"]:
            df = pd.DataFrame(progress_data["sessions"])

            # Convert date strings to datetime if needed
            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])

            # Resample by week
            df_weekly = df.resample('W', on='date').agg({
                'duration': 'sum',
                'calories': 'sum',
                'accuracy': 'mean'
            }).reset_index()

            # Create tabs for different views
            tab1, tab2 = st.tabs(["Duration", "Accuracy"])

            with tab1:
                st.area_chart(
                    df_weekly,
                    x="date",
                    y="duration",
                    color="#4CAF50",
                    height=300,
                    use_container_width=True
                )

            with tab2:
                st.line_chart(
                    df_weekly,
                    x="date",
                    y="accuracy",
                    color="#2196F3",
                    height=300,
                    use_container_width=True
                )
        else:
            st.info("🌟 Start your first session to see progress data!")

        # Pose-Specific Progress
        st.subheader("Pose Mastery")
        if progress_data["pose_accuracy"]:
            poses = sorted(progress_data["pose_accuracy"].keys())

            # Create expandable sections for each pose
            for pose in poses:
                accuracies = progress_data["pose_accuracy"][pose]
                avg_accuracy = sum(accuracies) / len(accuracies)
                last_accuracy = accuracies[-1]

                with st.expander(f"{pose.title()} ({len(accuracies)} sessions)"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Average", f"{avg_accuracy:.1f}%")
                        st.metric("Last", f"{last_accuracy:.1f}%")

                    with col2:
                        # Progress trend
                        df_pose = pd.DataFrame({
                            "date": [s["date"] for s in progress_data["sessions"] if s["pose"] == pose],
                            "accuracy": [s["accuracy"] for s in progress_data["sessions"] if s["pose"] == pose]
                        })
                        if not df_pose.empty:
                            st.line_chart(
                                df_pose.set_index("date"),
                                height=150,
                                use_container_width=True
                            )
        else:
            st.info("Practice different poses to track your mastery")

    except Exception as e:
        st.error(f"Error displaying progress: {str(e)}")


# ====================== TESTING ======================
if __name__ == "__main__":
    # Test data - uncomment to initialize
    # test_data = {
    #     "duration": 15,
    #     "calories": 120,
    #     "pose": "warrior",
    #     "accuracy": 85
    # }
    # update_progress(test_data)

    show_progress_main()
    show_progress_sidebar()