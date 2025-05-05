import streamlit as st
import pygame
import time
import os
from threading import Thread


class YogaNidraPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.audio_path = "assets/yoga_nidra_guide.mp3"
        self.is_playing = False
        self.paused = False
        self.current_position = 0
        self.total_duration = 941  # 21 minutes in seconds

        # Segment markers (in seconds)
        self.markers = {
            'intro': (0, 30),
            'body_scan': (30, 300),
            'breath_awareness': (300, 480),
            'visualization': (480, 720),
            'conclusion': (720, 1260)
        }

    def play(self):
        if not self.is_playing:
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play(start=self.current_position)
            self.is_playing = True
            self.paused = False
            Thread(target=self._update_progress, daemon=True).start()
        elif self.paused:
            pygame.mixer.music.unpause()
            self.paused = False

    def pause(self):
        if self.is_playing and not self.paused:
            self.current_position = pygame.mixer.music.get_pos() / 1000
            pygame.mixer.music.pause()
            self.paused = True

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.paused = False
        self.current_position = 0

    def set_volume(self, volume):
        pygame.mixer.music.set_volume(volume)

    def _update_progress(self):
        while self.is_playing:
            if not self.paused:
                self.current_position = pygame.mixer.music.get_pos() / 1000
            time.sleep(0.1)


def show_yoga_nidra_interface():
    """Streamlit UI for Yoga Nidra"""
    st.header("🌙 Yoga Nidra - Guided Meditation")

    # Initialize all session state variables
    if 'player' not in st.session_state:
        st.session_state.player = YogaNidraPlayer()
        st.session_state.volume = 0.7  # Default volume

    player = st.session_state.player

    # Create layout
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Play"):
            player.play()

    with col2:
        if st.button("⏸ Pause"):
            player.pause()

    with col3:
        if st.button("⏹ Stop"):
            player.stop()

    # Volume control (now properly initialized)
    volume = st.slider(
        "Volume",
        0.0,
        1.0,
        st.session_state.volume,  # Use initialized value
        key="volume_slider",
        on_change=lambda: player.set_volume(st.session_state.volume_slider))

    # Progress bar
    progress = player.current_position / player.total_duration if player.total_duration > 0 else 0
    st.progress(progress)

    # Session timer
    current_time = time.strftime('%M:%S', time.gmtime(player.current_position))
    total_time = time.strftime('%M:%S', time.gmtime(player.total_duration))
    st.caption(f"⏱️ {current_time} / {total_time}")

    # Session segments
    st.subheader("Session Segments")
    for segment, (start, end) in player.markers.items():
        st.write(f"- {segment.replace('_', ' ').title()}: {start // 60}:{start % 60:02d} - {end // 60}:{end % 60:02d}")

    # Instructions
    st.markdown("""
    ### How to Use:
    1. Lie down in a comfortable position
    2. Use headphones for best experience
    3. Follow the guided instructions
    4. Don't fall asleep completely!
    """)

    # Set initial volume
    player.set_volume(st.session_state.volume)