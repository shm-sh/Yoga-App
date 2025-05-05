import streamlit as st
import os
from PIL import Image


# ====================== PATH HELPER ======================
def get_image_path(image_rel_path):
    """Convert relative path to absolute path with verification"""
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Script's directory
    full_path = os.path.join(base_dir, image_rel_path)

    # Debugging: Check if path exists
    if not os.path.exists(full_path):
        st.error(f"Path doesn't exist: {full_path}")
    return full_path

# ====================== POSE DATABASE ======================
POSE_LIBRARY = {
    "Beginner": [
        {
            "name": "Mountain (Tadasana)",
            "image": "C:\\Users\\shaur\\OneDrive\\Desktop\\yoga_app\\yoga_pose\\beginner\\mountain.jpg",
            "benefits": "Improves posture, strengthens thighs",
            "instructions": ["Stand tall", "Feet hip-width apart", "Arms by sides"]
        },
        {
            "name": "Child's Pose (Balasana)",
            "image": get_image_path("yoga_pose/beginner/child_pose.jpg"),
            "benefits": "Relaxes spine, relieves stress",
            "instructions": ["Kneel on floor", "Sit on heels", "Stretch arms forward"]
        },
        {
            "name": "Downward-Facing Dog (Adho Mukha Svanasana)",
            "image": "yoga_pose/beginner/downward_dog.jpg",
            "benefits": "Stretches hamstrings, strengthens arms",
            "instructions": ["Start on hands and knees", "Lift hips up", "Heels toward floor"]
        },
        {
            "name": "Warrior I (Virabhadrasana I)",
            "image": "yoga_pose/beginner/warrior1.jpg",
            "benefits": "Strengthens legs, opens hips",
            "instructions": ["Step one foot back", "Bend front knee", "Arms overhead"]
        },
        {
            "name": "Warrior II (Virabhadrasana II)",
            "image": "yoga_pose/beginner/warrior2.jpg",
            "benefits": "Improves balance, tones legs",
            "instructions": ["Feet wide apart", "Arms parallel to floor", "Gaze over front hand"]
        },
        {
            "name": "Tree Pose (Vrksasana)",
            "image": "yoga_pose/beginner/tree_pose.jpg",
            "benefits": "Enhances balance, strengthens ankles",
            "instructions": ["Stand on one leg", "Place foot on inner thigh", "Hands in prayer"]
        },
        {
            "name": "Cat-Cow Stretch (Marjaryasana-Bitilasana)",
            "image": "yoga_pose/beginner/cat_cow.jpg",
            "benefits": "Improves spine flexibility",
            "instructions": ["Arch back (cat)", "Drop belly (cow)", "Sync with breath"]
        },
        {
            "name": "Cobra Pose (Bhujangasana)",
            "image": "yoga_pose/beginner/cobra.jpg",
            "benefits": "Strengthens back, opens chest",
            "instructions": ["Lie on stomach", "Press into hands", "Lift chest"]
        },
        {
            "name": "Bridge Pose (Setu Bandhasana)",
            "image": "yoga_pose/beginner/bridge.jpg",
            "benefits": "Strengthens glutes, stretches chest",
            "instructions": ["Lie on back", "Feet flat, knees bent", "Lift hips up"]
        },
        {
            "name": "Seated Forward Bend (Paschimottanasana)",
            "image": "yoga_pose/beginner/seated_forward_bend.jpg",
            "benefits": "Stretches hamstrings, calms mind",
            "instructions": ["Sit legs straight", "Hinge at hips", "Reach toward toes"]
        },
        {
            "name": "Easy Pose (Sukhasana)",
            "image": "yoga_pose/beginner/easy_pose.jpg",
            "benefits": "Promotes relaxation, aligns spine",
            "instructions": ["Sit cross-legged", "Hands on knees", "Spine tall"]
        },
        {
            "name": "Corpse Pose (Savasana)",
            "image": "yoga_pose/beginner/corpse_pose.jpg",
            "benefits": "Deep relaxation, reduces stress",
            "instructions": ["Lie flat on back", "Arms by sides", "Close eyes"]
        },
        {
            "name": "Standing Forward Bend (Uttanasana)",
            "image": "yoga_pose/beginner/standing_forward_bend.jpg",
            "benefits": "Stretches hamstrings, relieves tension",
            "instructions": ["Fold at hips", "Let head hang", "Bend knees slightly"]
        },
        {
            "name": "Triangle Pose (Trikonasana)",
            "image": "yoga_pose/beginner/triangle.jpg",
            "benefits": "Stretches sides, improves digestion",
            "instructions": ["Feet wide apart", "Reach to shin or floor", "Arm up"]
        },
        {
            "name": "Happy Baby (Ananda Balasana)",
            "image": "yoga_pose/beginner/happy_baby.jpg",
            "benefits": "Relieves lower back tension",
            "instructions": ["Lie on back", "Hold feet", "Rock gently"]
        },
        {
            "name": "Reclining Butterfly (Supta Baddha Konasana)",
            "image": "yoga_pose/beginner/reclining_butterfly.jpg",
            "benefits": "Opens hips, relaxes body",
            "instructions": ["Lie on back", "Soles of feet together", "Knees drop open"]
        },
        {
            "name": "Legs-Up-the-Wall (Viparita Karani)",
            "image": "yoga_pose/beginner/legs_up_wall.jpg",
            "benefits": "Relieves tired legs, calms mind",
            "instructions": ["Sit close to wall", "Swing legs up", "Rest arms by sides"]
        }
    ],

    "Intermediate": [
        {
            "name": "Warrior III (Virabhadrasana III)",
            "image": "yoga_pose/intermediate/warrior3.jpg",
            "benefits": "Improves balance, strengthens core",
            "instructions": ["Stand on one leg", "Extend other leg back", "Arms forward, torso parallel to floor"]
        },
        {
            "name": "Half Moon (Ardha Chandrasana)",
            "image": "yoga_pose/intermediate/half_moon.jpg",
            "benefits": "Enhances coordination, stretches hamstrings",
            "instructions": ["From Triangle Pose", "Lift top leg", "Open hips, reach top arm up"]
        },
        {
            "name": "Crow Pose (Bakasana)",
            "image": "yoga_pose/intermediate/crow.jpg",
            "benefits": "Builds arm strength, improves focus",
            "instructions": ["Squat low", "Place knees on triceps", "Lean forward, lift feet"]
        },
        {
            "name": "Extended Side Angle (Utthita Parsvakonasana)",
            "image": "yoga_pose/intermediate/side_angle.jpg",
            "benefits": "Strengthens legs, opens chest",
            "instructions": ["Warrior II base", "Lower forearm to thigh", "Top arm reaches overhead"]
        },
        {
            "name": "Wheel Pose (Urdhva Dhanurasana)",
            "image": "yoga_pose/intermediate/wheel.jpg",
            "benefits": "Opens chest, strengthens back",
            "instructions": ["Lie on back", "Feet and hands planted", "Lift hips and chest"]
        },
        {
            "name": "Firefly Pose (Tittibhasana)",
            "image": "yoga_pose/intermediate/firefly.jpg",
            "benefits": "Strengthens wrists, core, and arms",
            "instructions": ["Squat with hands inside knees", "Shift weight to hands", "Straighten legs sideways"]
        },
        {
            "name": "King Pigeon (Eka Pada Rajakapotasana)",
            "image": "yoga_pose/intermediate/king_pigeon.jpg",
            "benefits": "Deep hip opener, stretches quadriceps",
            "instructions": ["From Pigeon Pose", "Bend back leg", "Reach for foot with hand"]
        },
        {
            "name": "Headstand (Sirsasana)",
            "image": "yoga_pose/intermediate/headstand.jpg",
            "benefits": "Boosts circulation, builds core strength",
            "instructions": ["Forearms on mat", "Crown of head down", "Lift legs slowly"]
        },
        {
            "name": "Forearm Stand (Pincha Mayurasana)",
            "image": "yoga_pose/intermediate/forearm_stand.jpg",
            "benefits": "Strengthens shoulders, improves balance",
            "instructions": ["Forearms parallel", "Walk feet in", "Kick up one leg at a time"]
        },
        {
            "name": "Camel Pose (Ustrasana)",
            "image": "yoga_pose/intermediate/camel.jpg",
            "benefits": "Stretches front body, improves posture",
            "instructions": ["Kneel with hips forward", "Reach for heels", "Arch back gently"]
        },
        {
            "name": "Side Crow (Parsva Bakasana)",
            "image": "yoga_pose/intermediate/side_crow.jpg",
            "benefits": "Twists spine, strengthens obliques",
            "instructions": ["Twist from Crow Pose", "Knees to one elbow", "Engage core"]
        },
        {
            "name": "Dancer’s Pose (Natarajasana)",
            "image": "yoga_pose/intermediate/dancer.jpg",
            "benefits": "Improves balance, stretches shoulders",
            "instructions": ["Stand on one leg", "Hold back foot", "Kick into hand, lean forward"]
        },
        {
            "name": "Eight-Angle Pose (Astavakrasana)",
            "image": "yoga_pose/intermediate/eight_angle.jpg",
            "benefits": "Strengthens arms, challenges balance",
            "instructions": ["Squat with legs hooked over arm", "Lean forward", "Lift feet off ground"]
        },
        {
            "name": "Flying Pigeon (Eka Pada Galavasana)",
            "image": "yoga_pose/intermediate/flying_pigeon.jpg",
            "benefits": "Opens hips, builds arm strength",
            "instructions": ["From Crow Pose", "Cross ankle over knee", "Lean forward, lift back leg"]
        },
        {
            "name": "Scorpion Pose (Vrschikasana)",
            "image": "yoga_pose/intermediate/scorpion.jpg",
            "benefits": "Deep backbend, strengthens shoulders",
            "instructions": ["From Forearm Stand", "Bend knees, reach feet toward head"]
        },
        {
            "name": "Revolved Triangle (Parivrtta Trikonasana)",
            "image": "yoga_pose/intermediate/revolved_triangle.jpg",
            "benefits": "Detoxifies, improves spinal mobility",
            "instructions": ["From Triangle Pose", "Twist torso", "Lower hand to floor or shin"]
        },
        {
            "name": "Peacock Pose (Mayurasana)",
            "image": "yoga_pose/intermediate/peacock.jpg",
            "benefits": "Strengthens wrists and core",
            "instructions": ["Kneel with hands turned back", "Lean forward", "Lift legs off ground"]
        }
    ],

    "Advanced": [
        {
            "name": "Headstand (Sirsasana)",
            "image": "yoga_pose/advanced/headstand.jpg",
            "benefits": "Improves circulation, strengthens core",
            "instructions": ["Interlace fingers", "Place crown on mat", "Lift legs up"],
            "warning": "Avoid if you have neck pain"
        },
        {
            "name": "Handstand (Adho Mukha Vrksasana)",
            "image": "yoga_pose/advanced/handstand.jpg",
            "benefits": "Builds shoulder strength, boosts confidence",
            "instructions": ["Kick up against a wall", "Engage core", "Stack hips over shoulders"],
            "warning": "Practice near a wall initially"
        },
        {
            "name": "Scorpion Pose (Vrschikasana)",
            "image": "yoga_pose/advanced/scorpion.jpg",
            "benefits": "Deep backbend, opens shoulders",
            "instructions": ["From forearm stand", "Bend knees", "Reach feet toward head"],
            "warning": "Requires advanced back flexibility"
        },
        {
            "name": "One-Legged King Pigeon (Eka Pada Rajakapotasana)",
            "image": "yoga_pose/advanced/king_pigeon.jpg",
            "benefits": "Extreme hip opener, stretches quadriceps",
            "instructions": ["From pigeon pose", "Bend back leg", "Grab foot with both hands"],
            "warning": "Avoid if you have knee injuries"
        },
        {
            "name": "Flying Splits (Eka Pada Koundinyasana II)",
            "image": "yoga_pose/advanced/flying_splits.jpg",
            "benefits": "Strengthens arms, improves balance",
            "instructions": ["From twisted lunge", "Hook leg over arm", "Lift back leg"],
            "warning": "Requires strong core engagement"
        },
        {
            "name": "Mermaid Pose (Eka Pada Sirsasana)",
            "image": "yoga_pose/advanced/mermaid.jpg",
            "benefits": "Deep stretch for hips and quadriceps",
            "instructions": ["From seated", "Bend one leg back", "Reach overhead for foot"],
            "warning": "Not for those with hip replacements"
        },
        {
            "name": "Eight-Angle Pose (Astavakrasana)",
            "image": "yoga_pose/advanced/eight_angle.jpg",
            "benefits": "Strengthens wrists and obliques",
            "instructions": ["Squat low", "Hook legs around arm", "Lean forward to lift"],
            "warning": "Avoid with wrist injuries"
        },
        {
            "name": "Tittibhasana (Firefly Pose)",
            "image": "yoga_pose/advanced/firefly.jpg",
            "benefits": "Develops arm and core strength",
            "instructions": ["Squat with hands inside knees", "Shift weight to hands", "Extend legs straight"],
            "warning": "Requires hamstring flexibility"
        },
        {
            "name": "Standing Splits (Urdhva Prasarita Eka Padasana)",
            "image": "yoga_pose/advanced/standing_splits.jpg",
            "benefits": "Improves hamstring flexibility and balance",
            "instructions": ["From forward fold", "Lift one leg high", "Square hips"],
            "warning": "Avoid if you have low back issues"
        },
        {
            "name": "Formidable Face Pose (Gandha Bherundasana)",
            "image": "yoga_pose/advanced/formidable_face.jpg",
            "benefits": "Extreme backbend, opens chest",
            "instructions": ["Lie on belly", "Grab ankles", "Lift thighs and chest"],
            "warning": "Advanced backbend—warm up thoroughly"
        },
        {
            "name": "Peacock Pose (Mayurasana)",
            "image": "yoga_pose/advanced/peacock.jpg",
            "benefits": "Strengthens wrists and core",
            "instructions": ["Kneel with hands turned back", "Lean forward", "Lift legs off ground"],
            "warning": "Avoid with wrist injuries"
        },
        {
            "name": "Lotus Headstand (Padma Sirsasana)",
            "image": "yoga_pose/advanced/lotus_headstand.jpg",
            "benefits": "Combines inversion and hip opening",
            "instructions": ["Enter lotus pose", "Place crown on mat", "Lift hips up"],
            "warning": "Advanced—master headstand and lotus first"
        },
        {
            "name": "One-Handed Handstand (Eka Hasta Vrksasana)",
            "image": "yoga_pose/advanced/one_hand_handstand.jpg",
            "benefits": "Builds extreme shoulder strength",
            "instructions": ["From handstand", "Shift weight to one hand", "Engage core"],
            "warning": "Only attempt after mastering handstand"
        },
        {
            "name": "Yogic Sleep Pose (Yoganidrasana)",
            "image": "yoga_pose/advanced/yogic_sleep.jpg",
            "benefits": "Deep hip and shoulder stretch",
            "instructions": ["Lie on back", "Thread legs behind head", "Bind hands"],
            "warning": "Extreme flexibility required"
        },
        {
            "name": "Dragonfly Pose (Maksikanagasana)",
            "image": "yoga_pose/advanced/dragonfly.jpg",
            "benefits": "Strengthens arms, challenges balance",
            "instructions": ["From crow pose", "Extend one leg back", "Straighten both arms"],
            "warning": "Requires open hips and strong core"
        },
        {
            "name": "Chin Stand (Galavasana)",
            "image": "yoga_pose/advanced/chin_stand.jpg",
            "benefits": "Deep backbend, strengthens shoulders",
            "instructions": ["From forearm stand", "Lower chin to mat", "Arch back"],
            "warning": "Avoid if you have neck issues"
        }
    ],

    "Surya Namaskar": [
        {
            "name": "1. Prayer Pose (Pranamasana)",
            "image": "yoga_pose/surya_namaskar/prayer.jpg",
            "benefits": "Centers the mind, prepares the body",
            "breath": "Exhale"
        },
        {
            "name": "2. Raised Arms Pose (Hasta Uttanasana)",
            "image": "yoga_pose/surya_namaskar/raised_arms.jpg",
            "benefits": "Stretches abdomen, expands lungs",
            "breath": "Inhale"
        },
        {
            "name": "3. Standing Forward Bend (Uttanasana)",
            "image": "yoga_pose/surya_namaskar/forward_bend.jpg",
            "benefits": "Stretches hamstrings, calms the mind",
            "breath": "Exhale"
        },
        {
            "name": "4. Half Standing Forward Bend (Ardha Uttanasana)",
            "image": "yoga_pose/surya_namaskar/half_forward_bend.jpg",
            "benefits": "Lengthens spine, engages core",
            "breath": "Inhale"
        },
        {
            "name": "5. Plank Pose (Phalakasana)",
            "image": "yoga_pose/surya_namaskar/plank.jpg",
            "benefits": "Strengthens arms, shoulders, and core",
            "breath": "Exhale"
        },
        {
            "name": "6. Eight-Limbed Pose (Ashtanga Namaskara)",
            "image": "yoga_pose/surya_namaskar/eight_limbed.jpg",
            "benefits": "Strengthens chest and legs",
            "breath": "Hold (or exhale)"
        },
        {
            "name": "7. Cobra Pose (Bhujangasana)",
            "image": "yoga_pose/surya_namaskar/cobra.jpg",
            "benefits": "Opens the chest, stretches the spine",
            "breath": "Inhale"
        },
        {
            "name": "8. Downward-Facing Dog (Adho Mukha Svanasana)",
            "image": "yoga_pose/surya_namaskar/downward_dog.jpg",
            "benefits": "Stretches hamstrings, relieves stress",
            "breath": "Exhale"
        },
        {
            "name": "9. Half Standing Forward Bend (Ardha Uttanasana)",
            "image": "yoga_pose/surya_namaskar/half_forward_bend.jpg",
            "benefits": "Prepares the body to stand",
            "breath": "Inhale"
        },
        {
            "name": "10. Standing Forward Bend (Uttanasana)",
            "image": "yoga_pose/surya_namaskar/forward_bend.jpg",
            "benefits": "Releases tension in the back",
            "breath": "Exhale"
        },
        {
            "name": "11. Raised Arms Pose (Hasta Uttanasana)",
            "image": "yoga_pose/surya_namaskar/raised_arms.jpg",
            "benefits": "Stretches the whole body",
            "breath": "Inhale"
        },
        {
            "name": "12. Prayer Pose (Pranamasana)",
            "image": "yoga_pose/surya_namaskar/prayer.jpg",
            "benefits": "Brings focus back to the center",
            "breath": "Exhale"
        }
    ]
}


# ====================== HELPER FUNCTIONS ======================
def load_pose_image(image_path):
    """Load pose image with error handling"""
    try:
        return Image.open(image_path)
    except Exception as e:
        st.warning(f"Image not found at: {str(e)}")
        return None

def show_pose_details(pose):
    """Display pose details in a card"""
    col1, col2 = st.columns([1, 2])

    with col1:
        img = load_pose_image(pose["image"])
        if img:
            st.image(img, use_container_width=True)

    with col2:
        st.subheader(pose["name"])
        st.markdown(f"**Benefits**: {pose['benefits']}")

        if "instructions" in pose:
            with st.expander("Step-by-step instructions"):
                for step in pose["instructions"]:
                    st.markdown(f"- {step}")

        if "warning" in pose:
            st.warning(pose["warning"])

        # ====================== MAIN UI ======================


def display_pose_library():
    st.title("🧘 Yoga Pose Library (50+ Poses)")

    # Tabbed interface for categories
    tab1, tab2, tab3, tab4 = st.tabs(["Beginner", "Intermediate", "Advanced", "Surya Namaskar"])

    # Beginner Poses
    with tab1:
        st.subheader("Beginner Poses (20 Poses)")
        for pose in POSE_LIBRARY["Beginner"]:
            show_pose_details(pose)
            st.divider()

            # Intermediate Poses
    with tab2:
        st.subheader("Intermediate Poses (20 Poses)")
        for pose in POSE_LIBRARY["Intermediate"]:
            show_pose_details(pose)
            st.divider()

            # Advanced Poses
    with tab3:
        st.subheader("Advanced Poses (10 Poses)")
        st.warning("⚠️ Requires experience! Consult a teacher.")
        for pose in POSE_LIBRARY["Advanced"]:
            show_pose_details(pose)
            st.divider()

            # Surya Namaskar
    with tab4:
        st.subheader("Surya Namaskar (12 Steps)")
        st.info("Traditional Sun Salutation sequence. Flow with your breath.")
        for pose in POSE_LIBRARY["Surya Namaskar"]:
            st.markdown(f"### {pose['name']}")
            show_pose_details(pose)
            st.markdown(f"**Breath**: {pose['breath']}")
            st.divider()

        # ====================== INTEGRATION ======================


def integrate_pose_library():
    """Call this in your main app"""
    display_pose_library()


if __name__ == "__main__":
    display_pose_library()