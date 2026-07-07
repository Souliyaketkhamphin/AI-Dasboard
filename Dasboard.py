import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
from datetime import datetime

# =========================================================
# N8N COUNSELOR ALERT SETTINGS
# =========================================================

# Keep private data in Streamlit Secrets, not in public GitHub code.
# Streamlit Cloud: Manage app → Settings → Secrets.
N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL", "")
COUNSELOR_EMAIL = st.secrets.get("COUNSELOR_EMAIL", "")

AUTO_COUNSELOR_LEVELS = ["Moderate", "Severe"]


def send_report_to_n8n(report_data):
    if not N8N_WEBHOOK_URL:
        st.warning(
            "ຍັງບໍ່ໄດ້ຕັ້ງ N8N_WEBHOOK_URL ໃນ Streamlit Secrets. "
            "ການທຳນາຍຍັງໃຊ້ໄດ້, ແຕ່ການສົ່ງໄປ n8n ຈະບໍ່ເຮັດວຽກ."
        )
        return False

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=report_data,
            timeout=15
        )

        if response.status_code in [200, 201]:
            return True

        st.error(f"ລະບົບ n8n ຕອບກັບຂໍ້ຜິດພາດ: {response.status_code}")
        st.write(response.text)
        return False

    except Exception as e:
        st.error(f"ບໍ່ສາມາດສົ່ງຜົນໄປຫາ n8n ໄດ້: {e}")
        return False


def make_conclusion(risk):
    if risk == "Severe":
        return (
            "ຜົນການຄັດກອງສະແດງວ່າ ນັກຮຽນມີຄວາມສ່ຽງສູງ. "
            "ຄວນໃຫ້ທີ່ປຶກສາຕິດຕາມ ແລະ ໃຫ້ການຊ່ວຍເຫຼືອໂດຍໄວ. "
            "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
        )

    if risk == "Moderate":
        return (
            "ຜົນການຄັດກອງສະແດງວ່າ ນັກຮຽນມີຄວາມສ່ຽງປານກາງ. "
            "ຄວນມີການຕິດຕາມ ແລະ ໃຫ້ຄຳປຶກສາເມື່ອຈຳເປັນ. "
            "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
        )

    return (
        "ຜົນການຄັດກອງສະແດງວ່າ ນັກຮຽນມີຄວາມສ່ຽງຕ່ຳ. "
        "ແຕ່ຍັງຄວນດູແລ ແລະ ຕິດຕາມສຸຂະພາບຈິດຢ່າງຕໍ່ເນື່ອງ. "
        "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
    )


def make_json_safe(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="ແດຊບອດຄັດກອງຄວາມສ່ຽງ",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .sub-text {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .main-title, .sub-text, h1, h2, h3, p, div, label, input, textarea {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
    }

    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
    }

    span[data-testid="stIconMaterial"],
    span[class*="material-symbols"],
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "depression_model_package.pkl",
    BASE_DIR / "saved_model" / "depression_model_package.pkl",
]


def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    st.error(
        "ບໍ່ພົບໄຟລ໌ໂມເດວ. "
        "ກະລຸນາໃສ່ depression_model_package.pkl "
        "ໃນ main folder ຫຼື saved_model folder."
    )
    st.write("Checked paths:")
    for path in MODEL_CANDIDATES:
        st.code(str(path))
    st.stop()


@st.cache_resource
def load_model_package():
    model_path = find_model_path()
    return joblib.load(model_path)


st.markdown(
    """
    <div class="main-title">
    ແດຊບອດ AI ສຳລັບຄັດກອງຄວາມສ່ຽງຂອງນັກຮຽນ
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">
    ລະບົບນີ້ໃຊ້ເພື່ອຄັດກອງເບື້ອງຕົ້ນ ແລະ ສົ່ງແຈ້ງເຕືອນໄປຫາທີ່ປຶກສາເມື່ອຈຳເປັນ.
    </div>
    """,
    unsafe_allow_html=True
)

model_package = load_model_package()

model = model_package["model"]
preprocess = model_package["preprocess"]
le = model_package["label_encoder"]
common_cols = model_package["common_cols"]
num_cols = model_package["num_cols"]
cat_cols = model_package["cat_cols"]
best_model_name = model_package["best_model_name"]


# =========================================================
# LAO DISPLAY MAPS
# =========================================================

risk_lao_map = {
    "Low": "ຕ່ຳ",
    "Moderate": "ປານກາງ",
    "Severe": "ສູງ"
}

option_lao_map = {
    "low": "ຕ່ຳ",
    "moderate": "ປານກາງ",
    "severe": "ສູງ",
    "high": "ສູງ",
    "very high": "ສູງຫຼາຍ",
    "very_high": "ສູງຫຼາຍ",
    "extreme": "ສູງຫຼາຍ",
    "medium": "ປານກາງ",
    "average": "ປານກາງ",
    "below average": "ຕ່ຳກວ່າປານກາງ",
    "below_average": "ຕ່ຳກວ່າປານກາງ",
    "above average": "ສູງກວ່າປານກາງ",
    "above_average": "ສູງກວ່າປານກາງ",
    "fair": "ພໍໃຊ້",
    "good": "ດີ",
    "very good": "ດີຫຼາຍ",
    "excellent": "ດີເລີດ",
    "poor": "ອ່ອນ",
    "a lot": "ຫຼາຍ",
    "a little": "ໜ້ອຍ",
    "none": "ບໍ່ມີ",
    "never": "ບໍ່ເຄີຍ",
    "rarely": "ບໍ່ຄ່ອຍ",
    "sometimes": "ບາງຄັ້ງ",
    "often": "ເລື້ອຍໆ",
    "almost every day": "ເກືອບທຸກມື້",
    "male": "ຊາຍ",
    "female": "ຍິງ",
    "nan": "ບໍ່ລະບຸ",
    "unknown": "ບໍ່ລະບຸ",
    "": "ບໍ່ລະບຸ"
}


def normalize_text(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def display_option(value):
    if value is None:
        return "ບໍ່ລະບຸ"

    value_text = str(value).strip()
    key = normalize_text(value_text)

    if key in option_lao_map:
        return option_lao_map[key]

    if value_text.replace(".", "", 1).isdigit():
        return value_text

    if "very" in key and "high" in key:
        return "ສູງຫຼາຍ"
    if "below" in key and "average" in key:
        return "ຕ່ຳກວ່າປານກາງ"
    if "above" in key and "average" in key:
        return "ສູງກວ່າປານກາງ"

    return "ຕົວເລືອກອື່ນ"


def display_label(col):
    col_text = str(col).strip()
    col_low = col_text.lower()

    label_map = {
        "Academic performance self": "ການປະເມີນຜົນການຮຽນຂອງຕົນເອງ",
        "Academic pressure": "ຄວາມກົດດັນດ້ານການຮຽນ",
        "Average grade": "ຜົນການຮຽນໂດຍສະເລ່ຍ",
        "Homework pressure": "ຄວາມກົດດັນຈາກວຽກບ້ານ",
        "Homework pressure3": "ຄວາມກົດດັນຈາກວຽກບ້ານ",
        "School level": "ລະດັບຊັ້ນຮຽນ",
        "Stress_schoolwork": "ຄວາມຄຽດຈາກການຮຽນ",
        "Study hours day": "ຊົ່ວໂມງຮຽນຕໍ່ມື້",
        "Study satisfaction": "ຄວາມພໍໃຈຕໍ່ການຮຽນ",

        "Aep_sad_most_days": "ຮູ້ສຶກເສົ້າເກືອບທຸກມື້",
        "Anx_hard_to_relax": "ຜ່ອນຄາຍຍາກ",
        "Anx_nervous": "ຮູ້ສຶກກັງວົນ",
        "Anx_restless": "ຮູ້ສຶກຢູ່ບໍ່ນິ່ງ",
        "Anx_sleep_problem": "ມີບັນຫາການນອນ",
        "Anx_something_bad": "ກັງວົນວ່າຈະເກີດສິ່ງບໍ່ດີ",
        "Anx_upset_easily": "ອາລົມເສຍງ່າຍ",
        "Anx_worry_many_things": "ກັງວົນຫຼາຍເລື່ອງ",
        "Anxiety severity": "ລະດັບຄວາມກັງວົນ",
        "Dep_hopeless": "ຮູ້ສຶກໝົດຫວັງ",
        "Dep_self_harm_thoughts": "ຄວາມຄິດທຳຮ້າຍຕົນເອງ",
        "Dep_tired_often": "ຮູ້ສຶກເມື່ອຍເລື້ອຍໆ",
        "Dep_lose_interest": "ບໍ່ສົນໃຈສິ່ງທີ່ເຄີຍມັກ",
        "Sad or hopeless freq": "ຄວາມຖີ່ຂອງອາການເສົ້າ ຫຼື ໝົດຫວັງ",
        "Stress_daily_tasks": "ຄວາມຄຽດຈາກວຽກປະຈຳວັນ",
        "Stress_pressure_do_well": "ຄວາມກົດດັນໃຫ້ເຮັດໄດ້ດີ",
        "Stress_level_general": "ລະດັບຄວາມຄຽດໂດຍລວມ",
        "Worry or tense freq": "ຄວາມຖີ່ຂອງຄວາມກັງວົນ",

        "Family support when stressed": "ການຊ່ວຍເຫຼືອຈາກຄອບຄົວເມື່ອຄຽດ",
        "Family support": "ການຊ່ວຍເຫຼືອຈາກຄອບຄົວ",
        "Friend support": "ການຊ່ວຍເຫຼືອຈາກໝູ່",
        "Social support": "ການຊ່ວຍເຫຼືອທາງສັງຄົມ",
        "Parent support": "ການຊ່ວຍເຫຼືອຈາກພໍ່ແມ່",

        "Financial stress": "ຄວາມຄຽດດ້ານການເງິນ",
        "Feel_calm": "ຮູ້ສຶກສະຫງົບໃຈ",
        "Life_happy": "ຄວາມສຸກໃນຊີວິດ",
        "Sleep hours": "ຊົ່ວໂມງນອນ",
        "Exercise": "ການອອກກຳລັງກາຍ",
        "Phone use": "ການໃຊ້ໂທລະສັບ",
        "Online time": "ເວລາຢູ່ອອນລາຍ",

        "Age": "ອາຍຸ",
        "Gender": "ເພດ"
    }

    if col_text in label_map:
        return label_map[col_text]

    if "academic" in col_low:
        return "ການຮຽນ"
    if "grade" in col_low:
        return "ຜົນການຮຽນ"
    if "homework" in col_low:
        return "ວຽກບ້ານ"
    if "schoolwork" in col_low:
        return "ວຽກຮຽນ"
    if "school" in col_low:
        return "ໂຮງຮຽນ"
    if "study" in col_low:
        return "ການຮຽນ"
    if "anx" in col_low or "worry" in col_low or "nervous" in col_low:
        return "ຄວາມກັງວົນ"
    if "sad" in col_low:
        return "ຄວາມເສົ້າ"
    if "hopeless" in col_low:
        return "ຄວາມໝົດຫວັງ"
    if "depress" in col_low or "dep" in col_low:
        return "ອາການຊຶມເສົ້າ"
    if "stress" in col_low:
        return "ຄວາມຄຽດ"
    if "pressure" in col_low:
        return "ຄວາມກົດດັນ"
    if "sleep" in col_low:
        return "ການນອນ"
    if "family" in col_low:
        return "ຄອບຄົວ"
    if "parent" in col_low:
        return "ພໍ່ແມ່"
    if "friend" in col_low:
        return "ໝູ່ເພື່ອນ"
    if "social" in col_low or "support" in col_low:
        return "ການຊ່ວຍເຫຼືອ"
    if "financial" in col_low or "money" in col_low:
        return "ການເງິນ"
    if "phone" in col_low:
        return "ການໃຊ້ໂທລະສັບ"
    if "online" in col_low:
        return "ການໃຊ້ອິນເຕີເນັດ"
    if "happy" in col_low:
        return "ຄວາມສຸກ"
    if "calm" in col_low:
        return "ຄວາມສະຫງົບໃຈ"
    if "age" in col_low:
        return "ອາຍຸ"
    if "gender" in col_low:
        return "ເພດ"

    return "ຄຳຖາມທົ່ວໄປ"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏠 Home")
st.sidebar.caption("ເມນູ")

page = st.sidebar.radio(
    "ເລືອກໜ້າ",
    [
        "ໜ້າຫຼັກ",
        "ຜົນຂອງໂມເດວ",
        "ທຳນາຍຄວາມສ່ຽງ",
        "ກ່ຽວກັບ"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "ໜ້າຫຼັກ":
    st.header("ພາບລວມຂອງແດຊບອດ")

    st.info(
        "ແດຊບອດນີ້ໄດ້ເຊື່ອມກັບໂມເດວ AI ທີ່ຝຶກແລ້ວ. "
        "ລະບົບຈະຊ່ວຍຄັດກອງຄວາມສ່ຽງ ແລະ ແຈ້ງທີ່ປຶກສາເມື່ອຜົນຢູ່ໃນລະດັບທີ່ຄວນຕິດຕາມ."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("ຄວາມແມ່ນຍຳໃນຊຸດທົດສອບ", "92.81%")
    col2.metric("ຄວາມແມ່ນຍຳແບບສົມດຸນ", "96.99%")
    col3.metric("ຄະແນນ F1 ໂດຍລວມ", "96.28%")

    st.subheader("ໂມເດວທີ່ກຳລັງໃຊ້")
    st.success("ໂມເດວຖືກໂຫຼດສຳເລັດແລ້ວ")

    model_data = pd.DataFrame({
        "ຕົວຊີ້ວັດ": [
            "ຄວາມແມ່ນຍຳໃນຊຸດທົດສອບ",
            "ຄວາມແມ່ນຍຳແບບສົມດຸນ",
            "ຄະແນນ F1 ໂດຍລວມ"
        ],
        "ຄະແນນ": [92.81, 96.99, 96.28]
    })

    st.subheader("ສະຫຼຸບຜົນໂມເດວ")

    st.bar_chart(
        data=model_data,
        x="ຕົວຊີ້ວັດ",
        y="ຄະແນນ"
    )


# =========================================================
# MODEL RESULTS PAGE
# =========================================================

elif page == "ຜົນຂອງໂມເດວ":
    st.header("ຜົນຂອງໂມເດວ")

    st.write("ໜ້ານີ້ສະແດງຜົນການທົດສອບໂມເດວກັບຂໍ້ມູນຈິງ.")

    st.subheader("ຕາຕະລາງຜົນການຈັດປະເພດ")

    results_data = {
        "ລະດັບຄວາມສ່ຽງ": ["ຕ່ຳ", "ປານກາງ", "ສູງ"],
        "ຄວາມແມ່ນຍຳເມື່ອທຳນາຍ": [0.95, 0.94, 0.97],
        "ການກວດພົບໄດ້ຖືກ": [0.96, 0.93, 0.98],
        "ຄະແນນ F1": [0.95, 0.94, 0.97]
    }

    st.dataframe(
        results_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("ສະຫຼຸບຜົນການທຳນາຍ")

    col1, col2, col3 = st.columns(3)

    col1.metric("ນັກຮຽນທົດສອບທັງໝົດ", "440")
    col2.metric("ທຳນາຍຖືກ", "408")
    col3.metric("ທຳນາຍຜິດ", "32")

    st.info("ໂມເດວທຳນາຍຖືກ 408 ຄົນ ຈາກນັກຮຽນທົດສອບຈິງ 440 ຄົນ.")


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "ທຳນາຍຄວາມສ່ຽງ":
    st.header("ທຳນາຍຄວາມສ່ຽງຂອງນັກຮຽນ")

    st.info(
        "ກະລຸນາຕອບຄຳຖາມຂ້າງລຸ່ມ. "
        "ຄ່າ 1 ໝາຍເຖິງຕ່ຳຫຼາຍ ແລະ ຄ່າ 5 ໝາຍເຖິງສູງຫຼາຍ."
    )

    user_input = {}
    cat_options = {}

    try:
        cat_pipeline = preprocess.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["oh"]

        for col, options in zip(cat_cols, onehot.categories_):
            cat_options[col] = [str(x) for x in options]
    except Exception:
        cat_options = {}

    academic_cols = []
    mental_cols = []
    support_cols = []
    lifestyle_cols = []
    other_cols = []

    for col in common_cols:
        col_low = col.lower()

        if any(k in col_low for k in ["academic", "school", "study", "grade", "exam", "homework", "schoolwork"]):
            academic_cols.append(col)

        elif any(k in col_low for k in ["family", "friend", "social", "support", "parent", "home"]):
            support_cols.append(col)

        elif any(k in col_low for k in ["sleep", "exercise", "phone", "online", "relax", "tired", "happy", "calm", "financial", "money"]):
            lifestyle_cols.append(col)

        elif any(k in col_low for k in ["anx", "sad", "depress", "worry", "nervous", "hopeless", "stress", "pressure", "tense"]):
            mental_cols.append(col)

        else:
            other_cols.append(col)

    def input_field(col):
        label = display_label(col)
        safe_key = f"input_{col}"

        if col in num_cols:
            return float(
                st.slider(
                    label,
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    help="1 = ຕ່ຳຫຼາຍ, 5 = ສູງຫຼາຍ",
                    key=safe_key
                )
            )

        options = cat_options.get(col, [])

        clean_options = []
        for option in options:
            option_text = str(option).strip()
            if normalize_text(option_text) not in ["nan", "none", ""]:
                clean_options.append(option)

        if len(clean_options) > 0:
            display_labels = []
            display_to_original = {}

            for i, option in enumerate(clean_options):
                label_text = display_option(option)

                if label_text == "ຕົວເລືອກອື່ນ":
                    label_text = f"ຕົວເລືອກ {i + 1}"

                if label_text in display_to_original:
                    label_text = f"{label_text} {i + 1}"

                display_labels.append(label_text)
                display_to_original[label_text] = option

            selected_label = st.selectbox(
                label,
                options=display_labels,
                index=0,
                key=safe_key
            )

            return display_to_original[selected_label]

        text_value = st.text_input(
            label,
            value="",
            placeholder="ບໍ່ລະບຸ",
            key=safe_key
        )

        return text_value if text_value else "Unknown"

    def show_group(title, cols):
        st.markdown(f"### {title}")

        with st.container(border=True):
            if len(cols) == 0:
                st.caption("ບໍ່ພົບຊ່ອງຂໍ້ມູນ")
            else:
                col_a, col_b = st.columns(2)

                for i, col in enumerate(cols):
                    with col_a if i % 2 == 0 else col_b:
                        user_input[col] = input_field(col)

    def format_form_answers_lao(input_dict):
        lines = []

        for key, value in input_dict.items():
            lao_key = display_label(key)
            lao_value = display_option(value)
            lines.append(f"{lao_key}: {lao_value}")

        return "\n".join(lines)

    st.subheader("ປ້ອນຂໍ້ມູນນັກຮຽນ")

    show_group("1. ຂໍ້ມູນດ້ານການຮຽນ", academic_cols)
    show_group("2. ສຸຂະພາບຈິດ ແລະ ຄວາມຄຽດ", mental_cols)
    show_group("3. ຄອບຄົວ ແລະ ການຊ່ວຍເຫຼືອ", support_cols)
    show_group("4. ວິຖີຊີວິດ", lifestyle_cols)
    show_group("5. ຂໍ້ມູນອື່ນໆ", other_cols)

    st.divider()

    st.subheader("ຂໍ້ມູນສຳລັບສົ່ງແຈ້ງເຕືອນ")

    student_name_for_report = st.text_input(
        "ຊື່ນັກຮຽນ ຫຼື ລະຫັດນັກຮຽນ",
        key="student_name_auto"
    )

    consent_alert = st.checkbox(
        "ຂ້ອຍຍິນຍອມໃຫ້ສົ່ງຜົນຄັດກອງນີ້ໄປຫາທີ່ປຶກສາ",
        key="consent_auto"
    )

    st.divider()

    if st.button("ທຳນາຍຄວາມສ່ຽງ", use_container_width=True):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[common_cols]

        for col in cat_cols:
            input_df[col] = input_df[col].astype(str)

        input_p = preprocess.transform(input_df)
        input_p = input_p.toarray() if hasattr(input_p, "toarray") else np.asarray(input_p)

        pred = model.predict(input_p)[0]
        proba = model.predict_proba(input_p)[0]

        risk = le.inverse_transform([pred])[0]

        class_probabilities = {
            str(class_name): round(float(prob) * 100, 2)
            for class_name, prob in zip(le.classes_, proba)
        }

        low_p = class_probabilities.get("Low", 0.0)
        moderate_p = class_probabilities.get("Moderate", 0.0)
        severe_p = class_probabilities.get("Severe", 0.0)
        confidence = max(low_p, moderate_p, severe_p)

        conclusion = make_conclusion(risk)

        safe_answers = {
            str(k): make_json_safe(v)
            for k, v in user_input.items()
        }

        risk_lao = risk_lao_map.get(risk, risk)

        st.subheader("ຜົນການທຳນາຍ")

        if risk == "Low":
            st.success(f"ລະດັບຄວາມສ່ຽງ: {risk_lao}")
        elif risk == "Moderate":
            st.warning(f"ລະດັບຄວາມສ່ຽງ: {risk_lao}")
        else:
            st.error(f"ລະດັບຄວາມສ່ຽງ: {risk_lao}")

        prob_display_df = pd.DataFrame({
            "ລະດັບຄວາມສ່ຽງ": [
                risk_lao_map.get(str(class_name), str(class_name))
                for class_name in le.classes_
            ],
            "ຄວາມເປັນໄປໄດ້ (%)": [
                round(float(prob) * 100, 2)
                for prob in proba
            ]
        })

        st.subheader("ຄວາມເປັນໄປໄດ້ຂອງແຕ່ລະລະດັບ")

        st.dataframe(
            prob_display_df,
            use_container_width=True,
            hide_index=True
        )

        chart_df = prob_display_df.rename(
            columns={
                "ລະດັບຄວາມສ່ຽງ": "ລະດັບ",
                "ຄວາມເປັນໄປໄດ້ (%)": "ຄະແນນ"
            }
        )

        st.bar_chart(
            data=chart_df,
            x="ລະດັບ",
            y="ຄະແນນ"
        )

        st.subheader("ສະຫຼຸບຜົນ")
        st.write(conclusion)

        st.caption(
            "ເຄື່ອງມືນີ້ໃຊ້ເພື່ອການຄັດກອງ ແລະ ການວິຈັຍເທົ່ານັ້ນ. "
            "ບໍ່ຄວນໃຊ້ແທນການປະເມີນຈາກຜູ້ຊ່ຽວຊານ."
        )

        send_to_counselor = risk in AUTO_COUNSELOR_LEVELS

        if send_to_counselor:

            if not student_name_for_report:
                st.warning("ກະລຸນາໃສ່ຊື່ ຫຼື ລະຫັດນັກຮຽນກ່ອນສົ່ງ")

            elif not consent_alert:
                st.warning("ກະລຸນາກົດຍິນຍອມກ່ອນສົ່ງການແຈ້ງເຕືອນ")

            else:
                report_data = {
                    "student_name": student_name_for_report,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "risk_level": risk,
                    "risk_level_lao": risk_lao,
                    "low_probability": low_p,
                    "moderate_probability": moderate_p,
                    "severe_probability": severe_p,
                    "confidence": confidence,
                    "best_model_name": best_model_name,
                    "conclusion": conclusion,
                    "counselor_email": COUNSELOR_EMAIL,
                    "send_to_counselor": True,
                    "form_answers": safe_answers,
                    "form_answers_text": format_form_answers_lao(safe_answers),
                    "alert_flow": "Counselor only",
                    "disclaimer": "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
                }

                sent = send_report_to_n8n(report_data)

                if sent:
                    st.success(
                        "ສົ່ງຜົນໄປຫາລະບົບ n8n ສຳເລັດແລ້ວ. "
                        "ທີ່ປຶກສາຈະໄດ້ຮັບການແຈ້ງເຕືອນ."
                    )
                else:
                    st.error("ສົ່ງຜົນໄປ n8n ບໍ່ສຳເລັດ")

        else:
            st.info(
                "ຜົນຄວາມສ່ຽງຕ່ຳ. "
                "ບໍ່ມີການສົ່ງອີເມວອັດຕະໂນມັດ."
            )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ກ່ຽວກັບ":
    st.header("ກ່ຽວກັບແດຊບອດນີ້")

    st.write(
        """
        ແດຊບອດນີ້ເຊື່ອມກັບໂມເດວ AI ທີ່ຝຶກແລ້ວ
        ເພື່ອຊ່ວຍຄັດກອງຄວາມສ່ຽງຂອງນັກຮຽນ.

        ລະບົບຈະຮັບຂໍ້ມູນຈາກແບບຟອມ,
        ປະມວນຜົນດ້ວຍໂມເດວ,
        ແລະ ສະແດງຜົນເປັນ 3 ລະດັບ:
        ຕ່ຳ, ປານກາງ, ແລະ ສູງ.

        ຖ້າຜົນຢູ່ໃນລະດັບປານກາງ ຫຼື ສູງ,
        ລະບົບຈະສົ່ງການແຈ້ງເຕືອນໄປຫາທີ່ປຶກສາຜ່ານ n8n.

        ຜົນຈາກລະບົບນີ້ແມ່ນການຄັດກອງເບື້ອງຕົ້ນ.
        ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ.
        """
    )
