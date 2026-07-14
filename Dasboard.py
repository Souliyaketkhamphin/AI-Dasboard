import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
from datetime import datetime

APP_VERSION = "ລຸ້ນ: ເມນູຂ້າງອ່ານງ່າຍ v20"

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
            "ຍັງບໍ່ໄດ້ຕັ້ງລະບົບສົ່ງແຈ້ງເຕືອນ. "
            "ການທຳນາຍຍັງໃຊ້ໄດ້, ແຕ່ການສົ່ງແຈ້ງເຕືອນຈະຍັງບໍ່ເຮັດວຽກ."
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

        st.error(f"ລະບົບແຈ້ງເຕືອນມີຂໍ້ຜິດພາດ: {response.status_code}")
        st.write(response.text)
        return False

    except Exception as e:
        st.error(f"ບໍ່ສາມາດສົ່ງຜົນໄປຫາລະບົບແຈ້ງເຕືອນໄດ້: {e}")
        return False


def make_conclusion(risk, form_mode="student"):
    is_general = form_mode == "general"
    person_word = "ຜູ້ຕອບ" if is_general else "ນັກຮຽນ"
    helper_word = "ຄົນໃກ້ຊິດ ຫຼື ຜູ້ໃຫ້ຄຳປຶກສາ" if is_general else "ຄູ, ພໍ່ແມ່ ຫຼື ທີ່ປຶກສາ"

    if risk == "Severe":
        return (
            f"ຜົນການຄັດກອງສະແດງວ່າ {person_word} ມີຄວາມສ່ຽງສູງ. "
            f"ຄວນຮີບຄຸຍກັບ {helper_word} ເພື່ອຮັບການຊ່ວຍເຫຼືອ. "
            "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
        )

    if risk == "Moderate":
        return (
            f"ຜົນການຄັດກອງສະແດງວ່າ {person_word} ມີຄວາມສ່ຽງປານກາງ. "
            "ຄວນເລີ່ມດູແລຕົນເອງຫຼາຍຂຶ້ນ ແລະ ລອງຄຸຍກັບຄົນທີ່ໄວ້ໃຈ. "
            "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
        )

    return (
        f"ຜົນການຄັດກອງສະແດງວ່າ {person_word} ມີຄວາມສ່ຽງຕ່ຳ. "
        "ແຕ່ຍັງຄວນດູແລການນອນ, ອາຫານ, ອາລົມ ແລະ ຄົນຮອບຂ້າງ. "
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
    page_title="ແອັບດູແລໃຈ",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #fff9f2 0%, #fefcff 35%, #f3fbff 100%);
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #22304a;
    }

    .sub-text {
        font-size: 18px;
        margin-bottom: 25px;
        color: #394b63;
    }

    .main-title, .sub-text, h1, h2, h3, p, div, label, input, textarea {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f4f7ff 0%, #eefaf7 100%);
    }

    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        font-family: 'Phetsarath OT', 'Times New Roman', serif !important;
        color: #243047 !important;
        -webkit-text-fill-color: #243047 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        background: rgba(255,255,255,0.88) !important;
        border: 1px solid #d8e6f8 !important;
        border-radius: 18px !important;
        padding: 10px 10px 12px 10px !important;
        margin-top: 10px !important;
        box-shadow: 0 8px 18px rgba(91, 124, 180, 0.10) !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        font-size: 17px !important;
        font-weight: 800 !important;
        color: #22304a !important;
        -webkit-text-fill-color: #22304a !important;
        margin-bottom: 8px !important;
        display: block !important;
        position: static !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #b8d7ff !important;
        border-radius: 16px !important;
        min-height: 52px !important;
        color: #243047 !important;
    }

    .friendly-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.96);
        border: 1px solid #e9eef8;
        box-shadow: 0 8px 22px rgba(101, 119, 168, 0.08);
        margin-bottom: 14px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.96);
        border: 1px solid #e9eef8;
        padding: 14px 16px;
        border-radius: 18px;
        box-shadow: 0 8px 18px rgba(101, 119, 168, 0.06);
    }

    div[data-testid="stAlert"] {
        border-radius: 18px;
    }

    div[data-baseweb="select"], div[data-baseweb="input"] {
        border-radius: 14px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #edf1f7;
        border-radius: 22px;
    }

    .hero-box {
        padding: 22px 24px;
        border-radius: 24px;
        background: linear-gradient(135deg, #fff5c8 0%, #ffe3f0 50%, #dff5ff 100%);
        border: 1px solid #f4df8c;
        box-shadow: 0 10px 24px rgba(190, 160, 63, 0.12);
        margin-bottom: 18px;
    }

    .soft-pill {
        display: inline-block;
        padding: 7px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 16px;
        margin-top: 8px;
    }

    .risk-low { background: #dff7e8; color: #136b3c; }
    .risk-moderate { background: #fff2c7; color: #8a5a00; }
    .risk-severe { background: #ffe0e0; color: #a11b1b; }

    .mini-title {
        font-size: 15px;
        color: #5d6b82;
        margin-bottom: 6px;
    }

    .big-number {
        font-size: 32px;
        font-weight: 700;
        color: #22304a;
    }

    .tip-box {
        padding: 14px 16px;
        border-radius: 18px;
        background: #f7fbff;
        border-left: 6px solid #6fb1ff;
        margin-bottom: 10px;
    }

    .danger-box {
        padding: 14px 16px;
        border-radius: 18px;
        background: #fff0f0;
        border-left: 6px solid #ff7b7b;
        margin-top: 8px;
    }



    /* ===== Mobile and tablet support ===== */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 4.5rem !important;
        max-width: 1180px !important;
    }

    .stApp,
    .stApp p,
    .stApp div,
    .stApp label,
    .stApp span,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp li {
        color: #243047 !important;
        -webkit-text-fill-color: #243047 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input,
    textarea {
        background-color: #ffffff !important;
        color: #243047 !important;
        -webkit-text-fill-color: #243047 !important;
        border: 1px solid #d8e6f8 !important;
        border-radius: 16px !important;
        min-height: 48px !important;
        box-shadow: 0 3px 10px rgba(120, 145, 190, 0.07) !important;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] > label,
    [data-testid="stTextInput"] label,
    [data-testid="stSlider"] label {
        position: static !important;
        display: block !important;
        color: #22304a !important;
        -webkit-text-fill-color: #22304a !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
        z-index: 3 !important;
        background: transparent !important;
    }

    [data-testid="stSelectbox"] {
        margin-top: 4px !important;
        margin-bottom: 14px !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] * {
        color: #243047 !important;
        -webkit-text-fill-color: #243047 !important;
    }

    [role="listbox"],
    [data-baseweb="popover"] div {
        background-color: #ffffff !important;
    }

    [role="option"] {
        background-color: #ffffff !important;
        color: #243047 !important;
        min-height: 42px !important;
        font-size: 16px !important;
    }

    [role="option"]:hover {
        background-color: #eaf5ff !important;
    }

    button[kind="primary"],
    button[kind="secondary"],
    .stButton button {
        min-height: 48px !important;
        border-radius: 16px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
    }

    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        background: rgba(255,255,255,0.75) !important;
        border-radius: 14px !important;
        padding: 6px 8px !important;
        margin-bottom: 4px !important;
    }



    /* ===== Big colorful form-type buttons ===== */
    .st-key-form_mode_choice [data-testid="stRadio"] > label {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #22304a !important;
        margin-bottom: 10px !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 14px !important;
        margin-top: 8px !important;
        margin-bottom: 14px !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label {
        min-height: 74px !important;
        border-radius: 22px !important;
        padding: 14px 16px !important;
        border: 2px solid #d8e8ff !important;
        background: linear-gradient(135deg, #ffffff 0%, #eef8ff 100%) !important;
        box-shadow: 0 8px 20px rgba(94, 139, 200, 0.14) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        cursor: pointer !important;
        transition: all 0.18s ease-in-out !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(1) {
        background: linear-gradient(135deg, #e6f2ff 0%, #dff7ff 100%) !important;
        border-color: #7fc4ff !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(2) {
        background: linear-gradient(135deg, #fff0d6 0%, #ffe5f1 100%) !important;
        border-color: #ffb76b !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 24px rgba(94, 139, 200, 0.22) !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label:has(input:checked) {
        border: 3px solid #ff4b5c !important;
        box-shadow: 0 0 0 4px rgba(255, 75, 92, 0.14), 0 12px 24px rgba(255, 75, 92, 0.18) !important;
        transform: scale(1.02) !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label p,
    .st-key-form_mode_choice [role="radiogroup"] label span,
    .st-key-form_mode_choice [role="radiogroup"] label div {
        font-size: 19px !important;
        font-weight: 800 !important;
        color: #20314f !important;
        -webkit-text-fill-color: #20314f !important;
        line-height: 1.35 !important;
        text-align: center !important;
    }

    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(1) p::before,
    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(1) span::before {
        content: "🎒 ";
    }

    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(2) p::before,
    .st-key-form_mode_choice [role="radiogroup"] label:nth-child(2) span::before {
        content: "👥 ";
    }

    [data-testid="stSlider"] * {
        color: #243047 !important;
        -webkit-text-fill-color: #243047 !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.8rem !important;
            max-width: 100% !important;
        }

        .main-title {
            font-size: 23px !important;
            line-height: 1.45 !important;
            margin-bottom: 8px !important;
        }

        .sub-text {
            font-size: 15px !important;
            line-height: 1.6 !important;
            margin-bottom: 16px !important;
        }

        h1 { font-size: 25px !important; line-height: 1.45 !important; }
        h2 { font-size: 22px !important; line-height: 1.45 !important; }
        h3 { font-size: 19px !important; line-height: 1.5 !important; }
        p, label, div, span { font-size: 16px !important; line-height: 1.65 !important; }

        .hero-box,
        .friendly-card,
        .tip-box,
        .danger-box {
            padding: 14px 15px !important;
            border-radius: 18px !important;
            margin-bottom: 12px !important;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input,
        textarea {
            min-height: 52px !important;
            font-size: 17px !important;
            width: 100% !important;
        }



        .st-key-form_mode_choice [role="radiogroup"] {
            grid-template-columns: 1fr !important;
            gap: 12px !important;
        }

        .st-key-form_mode_choice [role="radiogroup"] label {
            min-height: 66px !important;
            padding: 12px 14px !important;
            border-radius: 20px !important;
        }

        .st-key-form_mode_choice [role="radiogroup"] label p,
        .st-key-form_mode_choice [role="radiogroup"] label span,
        .st-key-form_mode_choice [role="radiogroup"] label div {
            font-size: 18px !important;
        }

        [data-testid="stSidebar"] {
            min-width: 82vw !important;
            max-width: 82vw !important;
            background: #f4fbff !important;
        }

        [data-testid="stSidebar"] * {
            color: #243047 !important;
            -webkit-text-fill-color: #243047 !important;
        }

        .stDataFrame,
        [data-testid="stTable"] {
            overflow-x: auto !important;
        }
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
        "ກະລຸນາໃສ່ໄຟລ໌ໂມເດວໃຫ້ຖືກບ່ອນ. "
        "ໄຟລ໌ຕ້ອງຢູ່ໃນໂຟນເດີຫຼັກ ຫຼື ໂຟນເດີໂມເດວ."
    )
    st.write("ກະລຸນາກວດບ່ອນເກັບໄຟລ໌ໂມເດວອີກຄັ້ງ.")
    st.stop()


@st.cache_resource
def load_model_package():
    model_path = find_model_path()
    return joblib.load(model_path)


st.markdown(
    """
    <div class="main-title">
    ແອັບດູແລໃຈ ແລະ ຄັດກອງຄວາມສ່ຽງເບື້ອງຕົ້ນ
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">
    ໃຊ້ໄດ້ທັງນັກຮຽນ ແລະ ຄົນທົ່ວໄປ. ຜົນນີ້ເປັນການຄັດກອງເບື້ອງຕົ້ນ ບໍ່ແມ່ນການວິນິດໄສ.
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
    "very low": "ຕ່ຳຫຼາຍ",
    "moderate": "ປານກາງ",
    "medium": "ປານກາງ",
    "average": "ປານກາງ",
    "high": "ສູງ",
    "very high": "ສູງຫຼາຍ",
    "severe": "ສູງ",
    "extreme": "ສູງຫຼາຍ",
    "good": "ດີ",
    "very good": "ດີຫຼາຍ",
    "excellent": "ດີຫຼາຍ",
    "fair": "ປານກາງ",
    "poor": "ອ່ອນ",
    "yes": "ແມ່ນ",
    "no": "ບໍ່ແມ່ນ",
    "male": "ຊາຍ",
    "female": "ຍິງ",
    "never": "ບໍ່ເຄີຍ",
    "rarely": "ບໍ່ຄ່ອຍ",
    "sometimes": "ບາງຄັ້ງ",
    "often": "ເລື້ອຍໆ",
    "almost every day": "ເກືອບທຸກມື້",
    "always": "ສະເໝີ",
    "none": "ບໍ່ມີ",
    "nan": "ບໍ່ລະບຸ",
    "unknown": "ບໍ່ລະບຸ",
    "": "ບໍ່ລະບຸ"
}

# Exact form labels checked against your Google Form screenshots and your column list.
# ຂໍ້ຄວາມທີ່ຜູ້ໃຊ້ເຫັນໃຊ້ພາສາລາວແບບງ່າຍ.
LABEL_MAP = {
    # ຂໍ້ມູນພື້ນຖານ
    "Age": "ອາຍຸ",
    "Gender": "ເພດ",
    "Province": "ແຂວງທີ່ອາໄສຢູ່",
    "School level": "ລະດັບຊັ້ນຮຽນ",
    "Living_with": "ອາໄສຢູ່ກັບໃຜ",
    "Family_financial_status": "ຖານະການເງິນຂອງຄອບຄົວ",

    # ການຮຽນ
    "Average grade": "ຄະແນນສະເລ່ຍເທີມນີ້",
    "Academic performance self": "ຄິດວ່າການຮຽນຂອງຕົນເປັນແນວໃດ",
    "Focus in class": "ຕັ້ງໃຈຮຽນໃນຫ້ອງໄດ້ງ່າຍປານໃດ",
    "Academic pressure": "ຄວາມກົດດັນດ້ານການຮຽນ",
    "Homework pressure": "ຄວາມກົດດັນຈາກວຽກບ້ານ",
    "Homework pressure3": "ຄວາມກົດດັນຈາກວຽກບ້ານອີກຂໍ້ໜຶ່ງ",
    "CGPA": "ຄະແນນສະສົມ",
    "Study satisfaction": "ຄວາມພໍໃຈຕໍ່ການຮຽນ",
    "Sleep hours day": "ນອນຈັກຊົ່ວໂມງຕໍ່ມື້",
    "Study hours day": "ຮຽນຈັກຊົ່ວໂມງຕໍ່ມື້",

    # ວິຖີຊີວິດ
    "Sleep hours night": "ນອນຕອນກາງຄືນຈັກຊົ່ວໂມງ",
    "Healthy_meals_freq": "ກິນອາຫານດີຕໍ່ສຸຂະພາບເລື້ອຍປານໃດ",
    "Exercise_freq": "ອອກກຳລັງກາຍເລື້ອຍປານໃດ",
    "Online_time_daily": "ໃຊ້ເວລາອອນລາຍຕໍ່ມື້ເທົ່າໃດ",
    "Daytime_tiredness": "ຮູ້ສຶກເມື່ອຍຕອນກາງເວັນເລື້ອຍປານໃດ",
    "Skip_meals_freq": "ຂ້າມອາຫານເລື້ອຍປານໃດ",
    "Time_to_relax": "ມີເວລາພັກຜ່ອນຫຼັງເລີກຮຽນບໍ່",
    "Phone_before_sleep": "ໃຊ້ໂທລະສັບກ່ອນນອນດົນປານໃດ",

    # ສຸຂະພາບຈິດ ແລະ ຄວາມຄຽດ
    "Stress_level_general": "ຄວາມຄຽດໃນ 7 ມື້ຜ່ານມາ",
    "Worry or tense freq": "ຮູ້ສຶກກັງວົນ ຫຼື ຕຶງຄຽດເລື້ອຍປານໃດ",
    "Sad or hopeless freq": "ຮູ້ສຶກເສົ້າ ຫຼື ໝົດຫວັງເລື້ອຍປານໃດ",
    "Financial stress": "ຄວາມຄຽດເລື່ອງເງິນ",
    "Depression severity": "ລະດັບອາການເສົ້າໃນໃຈ",
    "Anxiety severity": "ລະດັບຄວາມກັງວົນ",

    # ຄວາມກັງວົນ
    "Anx_nervous": "ຮູ້ສຶກກັງວົນ ຫຼື ຢູ່ບໍ່ສະຫງົບ",
    "Anx_worry_many_things": "ກັງວົນຫຼາຍເລື່ອງ",
    "Anx_hard_to_relax": "ຜ່ອນຄາຍຍາກ",
    "Anx_restless": "ຢູ່ນິ່ງຍາກ",
    "Anx_upset_easily": "ອາລົມເສຍງ່າຍ",
    "Anx_something_bad": "ຮູ້ສຶກວ່າອາດມີສິ່ງບໍ່ດີເກີດຂຶ້ນ",
    "Anx_sleep_problem": "ກັງວົນຈົນນອນຍາກ",

    # ອາການເສົ້າໃນໃຈ
    "Aep_sad_most_days": "ຮູ້ສຶກເສົ້າເກືອບທຸກມື້",
    "Dep_lose_interest": "ບໍ່ສົນໃຈສິ່ງທີ່ເຄີຍມັກ",
    "Dep_tired_often": "ຮູ້ສຶກເມື່ອຍເລື້ອຍໆ",
    "Dep_sleep_problem": "ນອນໜ້ອຍ ຫຼື ຫຼາຍເກີນໄປ",
    "Dep_feel_failure": "ຮູ້ສຶກວ່າຕົນເອງລົ້ມເຫຼວ",
    "Dep_cannot_focus": "ຕັ້ງໃຈກັບວຽກຮຽນບໍ່ໄດ້",
    "Dep_slow_thinking": "ຄິດ ຫຼື ເຄື່ອນໄຫວຊ້າ",
    "Dep_hopeless": "ຮູ້ສຶກໝົດຫວັງ",
    "Dep_self_harm_thoughts": "ຄິດທຳຮ້າຍຕົນເອງ",

    # ຄວາມຄຽດ
    "Stress_schoolwork": "ວຽກຮຽນເຮັດໃຫ້ຄຽດ",
    "Stress_too_much_manage": "ມີຫຼາຍຢ່າງເກີນໄປທີ່ຕ້ອງຈັດການ",
    "Stress_daily_tasks": "ວຽກປະຈຳວັນເຮັດໃຫ້ໜັກໃຈ",
    "Stress_pressure_do_well": "ຮູ້ສຶກກົດດັນໃຫ້ເຮັດໄດ້ດີ",
    "Stress_hard_balance": "ຈັດເວລາຮຽນ ແລະ ຊີວິດຍາກ",

    # ຄອບຄົວ ແລະ ການຊ່ວຍເຫຼືອ
    "Family support when stressed": "ຄອບຄົວຊ່ວຍເຫຼືອເມື່ອຄຽດຫຼາຍປານໃດ",
    "Talk to someone when stressed": "ລົມກັບຄົນອື່ນເມື່ອຄຽດເລື້ອຍປານໃດ",
    "Support_someone_to_talk": "ມີຄົນໃຫ້ລົມດ້ວຍ",
    "Support_family": "ຄອບຄົວຊ່ວຍເຫຼືອຂ້ອຍ",
    "Support_friends": "ໝູ່ເພື່ອນຮັບຟັງຂ້ອຍ",
    "Support_understood": "ຄົນອື່ນເຂົ້າໃຈຂ້ອຍ",
    "Support_trust_adult": "ມີຜູ້ໃຫຍ່ 1 ຄົນທີ່ຂ້ອຍໄວ້ໃຈໄດ້",

    # ຄວາມສຸກ ແລະ ກຳລັງໃຈ
    "Life_happy": "ມີຄວາມສຸກກັບຊີວິດ",
    "Self_confident": "ຮູ້ສຶກໝັ້ນໃຈ",
    "Feel_calm": "ຮູ້ສຶກສະຫງົບໃຈ",
    "Handle_problems": "ຈັດການບັນຫາໄດ້",
    "Future_hope": "ມີຄວາມຫວັງຕໍ່ອະນາຄົດ",
}


# Column-level choices checked against the Google Form screenshots.
# The original value is still sent to the model.
# Only the text shown to the user is cleaned.
FORM_CHOICE_DISPLAY_MAP = {
    "Age": {
        "10 or lower": "10 ປີ ຫຼື ນ້ອຍກວ່າ",
        "under 10": "ນ້ອຍກວ່າ 10 ປີ",
        "10-11": "10–11 ປີ",
        "10–11": "10–11 ປີ",
        "12-14": "12–14 ປີ",
        "12–14": "12–14 ປີ",
        "15-17": "15–17 ປີ",
        "15–17": "15–17 ປີ",
        "18": "18 ປີ",
        "20-25": "20–25 ປີ",
        "20–25": "20–25 ປີ",
        "23-25": "23–25 ປີ",
        "23–25": "23–25 ປີ",
        "more than 30": "ຫຼາຍກວ່າ 30 ປີ",
    },
    "Province": {
        "phongsaly": "ຜົ້ງສາລີ",
        "luang namtha": "ຫຼວງນ້ຳທາ",
        "oudomxay": "ອຸດົມໄຊ",
        "xiengkhouang": "ຊຽງຂວາງ",
        "bokeo": "ບໍ່ແກ້ວ",
        "luang prabang": "ຫຼວງພະບາງ",
        "huaphan": "ຫົວພັນ",
        "xayaboury": "ໄຊຍະບູລີ",
        "sayaboury": "ໄຊຍະບູລີ",
        "vientiane capital": "ນະຄອນຫຼວງວຽງຈັນ",
        "vientiane province": "ແຂວງວຽງຈັນ",
        "borikhamxay": "ບໍລິຄຳໄຊ",
        "bolikhamxay": "ບໍລິຄຳໄຊ",
        "khammouane": "ຄຳມ່ວນ",
        "savannakhet": "ສະຫວັນນະເຂດ",
        "salavan": "ສາລະວັນ",
        "saravan": "ສາລະວັນ",
        "sekong": "ເຊກອງ",
        "champasak": "ຈຳປາສັກ",
        "attapeu": "ອັດຕະປື",
    },
    "School level": {
        "m.1": "ມ.1",
        "m1": "ມ.1",
        "ມ.1": "ມ.1",
        "m.2": "ມ.2",
        "m2": "ມ.2",
        "ມ.2": "ມ.2",
        "m.3": "ມ.3",
        "m3": "ມ.3",
        "ມ.3": "ມ.3",
        "m.4": "ມ.4",
        "m4": "ມ.4",
        "ມ.4": "ມ.4",
        "m.5": "ມ.5",
        "m5": "ມ.5",
        "ມ.5": "ມ.5",
        "m.6": "ມ.6",
        "m6": "ມ.6",
        "ມ.6": "ມ.6",
        "m.7": "ມ.7",
        "m7": "ມ.7",
        "ມ.7": "ມ.7",
    },
    "Gender": {
        "male": "ຊາຍ",
        "female": "ຍິງ",
    },
    "Average grade": {
        "80-100": "80–100 ຄະແນນ: ສູງ",
        "80–100": "80–100 ຄະແນນ: ສູງ",
        "60-79": "60–79 ຄະແນນ: ດີ",
        "60–79": "60–79 ຄະແນນ: ດີ",
        "40-59": "40–59 ຄະແນນ: ປານກາງ",
        "40–59": "40–59 ຄະແນນ: ປານກາງ",
        "below 40": "ຕ່ຳກວ່າ 40 ຄະແນນ",
    },
    "Academic performance self": {
        "very good": "ດີຫຼາຍ",
        "good": "ດີ",
        "average": "ປານກາງ",
        "below average": "ຕ່ຳກວ່າປານກາງ",
        "poor": "ອ່ອນ",
    },
    "Living_with": {
        "both parents": "ຢູ່ກັບພໍ່ແມ່ທັງສອງ",
        "one parent": "ຢູ່ກັບພໍ່ ຫຼື ແມ່ຄົນດຽວ",
        "relatives": "ຢູ່ກັບຍາດພີ່ນ້ອງ",
        "live alone": "ຢູ່ຄົນດຽວ",
    },
    "Family_financial_status": {
        "good": "ດີ",
        "fair": "ພໍໃຊ້",
        "poor": "ອ່ອນ",
        "very poor": "ອ່ອນຫຼາຍ",
    },
    "Sleep hours night": {
        "less than 5 hours": "ນ້ອຍກວ່າ 5 ຊົ່ວໂມງ",
        "5-6 hours": "5–6 ຊົ່ວໂມງ",
        "5–6 hours": "5–6 ຊົ່ວໂມງ",
        "7-8 hours": "7–8 ຊົ່ວໂມງ",
        "7–8 hours": "7–8 ຊົ່ວໂມງ",
        "more than 8 hours": "ຫຼາຍກວ່າ 8 ຊົ່ວໂມງ",
    },
    "Healthy_meals_freq": {
        "every day": "ທຸກມື້",
        "3-5 days/week": "3–5 ມື້ຕໍ່ອາທິດ",
        "3–5 days/week": "3–5 ມື້ຕໍ່ອາທິດ",
        "1-2 days/week": "1–2 ມື້ຕໍ່ອາທິດ",
        "1–2 days/week": "1–2 ມື້ຕໍ່ອາທິດ",
        "rarely": "ບໍ່ຄ່ອຍ",
    },
    "Exercise_freq": {
        "every day": "ທຸກມື້",
        "3-5 days/week": "3–5 ມື້ຕໍ່ອາທິດ",
        "3–5 days/week": "3–5 ມື້ຕໍ່ອາທິດ",
        "1-2 days/week": "1–2 ມື້ຕໍ່ອາທິດ",
        "1–2 days/week": "1–2 ມື້ຕໍ່ອາທິດ",
        "never": "ບໍ່ເຄີຍ",
    },
    "Online_time_daily": {
        "less than 1 hour": "ນ້ອຍກວ່າ 1 ຊົ່ວໂມງ",
        "1-3 hours": "1–3 ຊົ່ວໂມງ",
        "1–3 hours": "1–3 ຊົ່ວໂມງ",
        "4-6 hours": "4–6 ຊົ່ວໂມງ",
        "4–6 hours": "4–6 ຊົ່ວໂມງ",
        "more than 6 hours": "ຫຼາຍກວ່າ 6 ຊົ່ວໂມງ",
    },
    "Daytime_tiredness": {
        "never": "ບໍ່ເຄີຍ",
        "sometimes": "ບາງຄັ້ງ",
        "often": "ເລື້ອຍໆ",
        "almost every day": "ເກືອບທຸກມື້",
    },
    "Skip_meals_freq": {
        "never": "ບໍ່ເຄີຍ",
        "once a week": "1 ຄັ້ງຕໍ່ອາທິດ",
        "2-3 times/week": "2–3 ຄັ້ງຕໍ່ອາທິດ",
        "2–3 times/week": "2–3 ຄັ້ງຕໍ່ອາທິດ",
        "most days": "ເກືອບທຸກມື້",
    },
    "Time_to_relax": {
        "yes, every day": "ມີ ທຸກມື້",
        "a few days/week": "ມີ ບາງມື້ຕໍ່ອາທິດ",
        "rarely": "ບໍ່ຄ່ອຍມີ",
        "never": "ບໍ່ມີເລີຍ",
    },
    "Phone_before_sleep": {
        "never": "ບໍ່ເຄີຍ",
        "less than 30 minutes": "ນ້ອຍກວ່າ 30 ນາທີ",
        "30-60 minutes": "30–60 ນາທີ",
        "30–60 minutes": "30–60 ນາທີ",
        "more than 1 hour": "ຫຼາຍກວ່າ 1 ຊົ່ວໂມງ",
    },
    "Focus in class": {
        "very easy": "ງ່າຍຫຼາຍ",
        "easy": "ງ່າຍ",
        "hard": "ຍາກ",
        "very hard": "ຍາກຫຼາຍ",
    },
    "Worry or tense freq": {
        "never": "ບໍ່ເຄີຍ",
        "sometimes": "ບາງຄັ້ງ",
        "often": "ເລື້ອຍໆ",
        "almost every day": "ເກືອບທຸກມື້",
    },
    "Sad or hopeless freq": {
        "never": "ບໍ່ເຄີຍ",
        "sometimes": "ບາງຄັ້ງ",
        "often": "ເລື້ອຍໆ",
        "almost every day": "ເກືອບທຸກມື້",
    },
    "Family support when stressed": {
        "a lot": "ຫຼາຍ",
        "some": "ພໍມີ",
        "little": "ໜ້ອຍ",
        "none": "ບໍ່ມີ",
    },
    "Talk to someone when stressed": {
        "always": "ສະເໝີ",
        "sometimes": "ບາງຄັ້ງ",
        "rarely": "ບໍ່ຄ່ອຍ",
        "never": "ບໍ່ເຄີຍ",
    },
    "CGPA": {
        "0-5": "0–5: ຕ່ຳ",
        "0–5": "0–5: ຕ່ຳ",
        "5-7": "5–7: ປານກາງ",
        "5–7": "5–7: ປານກາງ",
        "7-8": "7–8: ສູງ",
        "7–8": "7–8: ສູງ",
        "8-10": "8–10: ສູງຫຼາຍ",
        "8–10": "8–10: ສູງຫຼາຍ",
    },
    "Study satisfaction": {
        "very low": "ຕ່ຳຫຼາຍ",
        "low": "ຕ່ຳ",
        "moderate": "ປານກາງ",
        "high": "ສູງ",
        "very high": "ສູງຫຼາຍ",
    },
}

# Labels for the general-people form.
# These keep the meaning broad and not school-only.
GENERAL_LABEL_MAP = {
    "School level": "ລະດັບການສຶກສາ",
    "Average grade": "ຜົນການຮຽນ ຫຼື ຜົນງານໂດຍລວມ",
    "Academic performance self": "ຄິດວ່າຜົນງານຂອງຕົນເປັນແນວໃດ",
    "Focus in class": "ຕັ້ງໃຈກັບວຽກ ຫຼື ກິດຈະກຳໄດ້ງ່າຍປານໃດ",
    "Academic pressure": "ຄວາມກົດດັນຈາກວຽກ ຫຼື ໜ້າທີ່",
    "Homework pressure": "ຄວາມກົດດັນຈາກວຽກທີ່ຕ້ອງເຮັດ",
    "Homework pressure3": "ວຽກຫຼາຍເຮັດໃຫ້ຮູ້ສຶກກົດດັນ",
    "CGPA": "ຜົນການຮຽນ ຫຼື ຜົນງານສະສົມ",
    "Study satisfaction": "ຄວາມພໍໃຈຕໍ່ວຽກ ຫຼື ການຮຽນຮູ້",
    "Study hours day": "ໃຊ້ເວລາຮຽນຮູ້ ຫຼື ເຮັດວຽກຈັກຊົ່ວໂມງຕໍ່ມື້",
    "Time_to_relax": "ມີເວລາພັກຜ່ອນບໍ່",
    "Stress_schoolwork": "ວຽກ ຫຼື ພາລະປະຈຳວັນເຮັດໃຫ້ຄຽດ",
    "Stress_hard_balance": "ຈັດເວລາວຽກ ແລະ ຊີວິດຍາກ",
    "Dep_cannot_focus": "ຕັ້ງໃຈກັບວຽກ ຫຼື ກິດຈະກຳບໍ່ໄດ້",
}

# Columns hidden from the general-people form.
# The model still receives safe neutral default values for these fields.
GENERAL_HIDDEN_COLUMNS = {
    "School level", "Average grade", "Academic performance self", "Focus in class",
    "Academic pressure", "Homework pressure", "Homework pressure3", "CGPA",
    "Study satisfaction", "Sleep hours day", "Study hours day", "Stress_schoolwork"
}

# Five form groups requested by the user.
# The basic student information now appears first.
SECTION_GROUPS = {
    "1. ຂໍ້ມູນພື້ນຖານ": [
        "Age", "Gender", "Province", "School level",
        "Living_with", "Family_financial_status"
    ],
    "2. ຂໍ້ມູນດ້ານການຮຽນ": [
        "Average grade", "Academic performance self", "Focus in class",
        "Academic pressure", "Homework pressure", "Homework pressure3",
        "CGPA", "Study satisfaction", "Sleep hours day", "Study hours day"
    ],
    "3. ວິຖີຊີວິດ": [
        "Sleep hours night", "Healthy_meals_freq", "Exercise_freq", "Online_time_daily",
        "Daytime_tiredness", "Skip_meals_freq", "Time_to_relax", "Phone_before_sleep"
    ],
    "4. ສຸຂະພາບຈິດ ແລະ ຄວາມຄຽດ": [
        "Stress_level_general", "Worry or tense freq", "Sad or hopeless freq",
        "Financial stress", "Depression severity", "Anxiety severity",
        "Anx_nervous", "Anx_worry_many_things", "Anx_hard_to_relax",
        "Anx_restless", "Anx_upset_easily", "Anx_something_bad", "Anx_sleep_problem",
        "Aep_sad_most_days", "Dep_lose_interest", "Dep_tired_often", "Dep_sleep_problem",
        "Dep_feel_failure", "Dep_cannot_focus", "Dep_slow_thinking", "Dep_hopeless",
        "Dep_self_harm_thoughts", "Stress_schoolwork", "Stress_too_much_manage",
        "Stress_daily_tasks", "Stress_pressure_do_well", "Stress_hard_balance",
        "Life_happy", "Self_confident", "Feel_calm", "Handle_problems", "Future_hope"
    ],
    "5. ຄອບຄົວ ແລະ ການຊ່ວຍເຫຼືອ": [
        "Family support when stressed", "Talk to someone when stressed",
        "Support_someone_to_talk", "Support_family", "Support_friends",
        "Support_understood", "Support_trust_adult"
    ],
    "6. ຂໍ້ມູນເພີ່ມເຕີມ": []
}

SECTION_ORDER = [
    "1. ຂໍ້ມູນພື້ນຖານ",
    "2. ຂໍ້ມູນດ້ານການຮຽນ",
    "3. ວິຖີຊີວິດ",
    "4. ສຸຂະພາບຈິດ ແລະ ຄວາມຄຽດ",
    "5. ຄອບຄົວ ແລະ ການຊ່ວຍເຫຼືອ",
    "6. ຂໍ້ມູນເພີ່ມເຕີມ"
]

GENERAL_SECTION_GROUPS = {
    "1. ຂໍ້ມູນທົ່ວໄປ": [
        "Age", "Gender", "Province", "Living_with", "Family_financial_status"
    ],
    "2. ວິຖີຊີວິດ": [
        "Sleep hours night", "Healthy_meals_freq", "Exercise_freq", "Online_time_daily",
        "Daytime_tiredness", "Skip_meals_freq", "Time_to_relax", "Phone_before_sleep"
    ],
    "3. ຄວາມຮູ້ສຶກ ແລະ ຄວາມຄຽດ": [
        "Stress_level_general", "Worry or tense freq", "Sad or hopeless freq",
        "Financial stress", "Depression severity", "Anxiety severity",
        "Anx_nervous", "Anx_worry_many_things", "Anx_hard_to_relax",
        "Anx_restless", "Anx_upset_easily", "Anx_something_bad", "Anx_sleep_problem",
        "Aep_sad_most_days", "Dep_lose_interest", "Dep_tired_often", "Dep_sleep_problem",
        "Dep_feel_failure", "Dep_cannot_focus", "Dep_slow_thinking", "Dep_hopeless",
        "Dep_self_harm_thoughts", "Stress_too_much_manage", "Stress_daily_tasks",
        "Stress_pressure_do_well", "Stress_hard_balance"
    ],
    "4. ການຊ່ວຍເຫຼືອ ແລະ ກຳລັງໃຈ": [
        "Family support when stressed", "Talk to someone when stressed",
        "Support_someone_to_talk", "Support_family", "Support_friends",
        "Support_understood", "Support_trust_adult",
        "Life_happy", "Self_confident", "Feel_calm", "Handle_problems", "Future_hope"
    ],
    "5. ຂໍ້ມູນເພີ່ມເຕີມ": []
}

GENERAL_SECTION_ORDER = [
    "1. ຂໍ້ມູນທົ່ວໄປ",
    "2. ວິຖີຊີວິດ",
    "3. ຄວາມຮູ້ສຶກ ແລະ ຄວາມຄຽດ",
    "4. ການຊ່ວຍເຫຼືອ ແລະ ກຳລັງໃຈ",
    "5. ຂໍ້ມູນເພີ່ມເຕີມ"
]

def normalize_text(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ").replace("–", " ")


def normalize_col_key(value):
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ").replace("–", " ")
    text = " ".join(text.split())
    return text


# Extra aliases for names that may be saved differently inside the model package.
COLUMN_ALIAS_MAP = {
    "age": "Age",
    "gender": "Gender",
    "province": "Province",
    "school level": "School level",
    "average grade": "Average grade",
    "academic performance self": "Academic performance self",
    "living with": "Living_with",
    "family financial status": "Family_financial_status",
    "sleep hours night": "Sleep hours night",
    "healthy meals freq": "Healthy_meals_freq",
    "exercise freq": "Exercise_freq",
    "online time daily": "Online_time_daily",
    "stress level general": "Stress_level_general",
    "daytime tiredness": "Daytime_tiredness",
    "skip meals freq": "Skip_meals_freq",
    "time to relax": "Time_to_relax",
    "phone before sleep": "Phone_before_sleep",
    "focus in class": "Focus in class",
    "worry or tense freq": "Worry or tense freq",
    "sad or hopeless freq": "Sad or hopeless freq",
    "family support when stressed": "Family support when stressed",
    "talk to someone when stressed": "Talk to someone when stressed",
    "academic pressure": "Academic pressure",
    "homework pressure": "Homework pressure",
    "homework pressure3": "Homework pressure3",
    "cgpa": "CGPA",
    "study satisfaction": "Study satisfaction",
    "sleep hours day": "Sleep hours day",
    "study hours day": "Study hours day",
    "financial stress": "Financial stress",
    "depression severity": "Depression severity",
    "anxiety severity": "Anxiety severity",
    "anx nervous": "Anx_nervous",
    "anx worry many things": "Anx_worry_many_things",
    "anx hard to relax": "Anx_hard_to_relax",
    "anx restless": "Anx_restless",
    "anx upset easily": "Anx_upset_easily",
    "anx something bad": "Anx_something_bad",
    "anx sleep problem": "Anx_sleep_problem",
    "aep sad most days": "Aep_sad_most_days",
    "dep lose interest": "Dep_lose_interest",
    "dep tired often": "Dep_tired_often",
    "dep sleep problem": "Dep_sleep_problem",
    "dep feel failure": "Dep_feel_failure",
    "dep cannot focus": "Dep_cannot_focus",
    "dep slow thinking": "Dep_slow_thinking",
    "dep hopeless": "Dep_hopeless",
    "dep self harm thoughts": "Dep_self_harm_thoughts",
    "stress schoolwork": "Stress_schoolwork",
    "stress too much manage": "Stress_too_much_manage",
    "stress daily tasks": "Stress_daily_tasks",
    "stress pressure do well": "Stress_pressure_do_well",
    "stress hard balance": "Stress_hard_balance",
    "support someone to talk": "Support_someone_to_talk",
    "support family": "Support_family",
    "support friends": "Support_friends",
    "support understood": "Support_understood",
    "support trust adult": "Support_trust_adult",
    "life happy": "Life_happy",
    "self confident": "Self_confident",
    "feel calm": "Feel_calm",
    "handle problems": "Handle_problems",
    "future hope": "Future_hope",
}


# If a model saved values as Option 1, Option 2, etc., show the real Lao choice.
ORDINAL_CHOICE_DISPLAY_MAP = {
    "Age": [
        "10 ປີ ຫຼື ນ້ອຍກວ່າ", "10–11 ປີ", "12–14 ປີ", "15–17 ປີ",
        "18–20 ປີ", "20–23 ປີ", "23–25 ປີ", "ຫຼາຍກວ່າ 30 ປີ"
    ],
    "Gender": ["ຊາຍ", "ຍິງ"],
    "School level": ["ມ.1", "ມ.2", "ມ.3", "ມ.4", "ມ.5", "ມ.6", "ມ.7"],
    "Province": [
        "ຜົ້ງສາລີ", "ຫຼວງນ້ຳທາ", "ອຸດົມໄຊ", "ຊຽງຂວາງ", "ບໍ່ແກ້ວ",
        "ຫຼວງພະບາງ", "ຫົວພັນ", "ໄຊຍະບູລີ", "ນະຄອນຫຼວງວຽງຈັນ",
        "ແຂວງວຽງຈັນ", "ບໍລິຄຳໄຊ", "ຄຳມ່ວນ", "ສະຫວັນນະເຂດ",
        "ສາລະວັນ", "ເຊກອງ", "ຈຳປາສັກ", "ອັດຕະປື"
    ],
    "Average grade": ["80–100 ຄະແນນ: ສູງ", "60–79 ຄະແນນ: ດີ", "40–59 ຄະແນນ: ປານກາງ", "ຕ່ຳກວ່າ 40 ຄະແນນ"],
    "Academic performance self": ["ດີຫຼາຍ", "ດີ", "ປານກາງ", "ຕ່ຳກວ່າປານກາງ", "ອ່ອນ"],
    "Living_with": ["ຢູ່ກັບພໍ່ແມ່ທັງສອງ", "ຢູ່ກັບພໍ່ ຫຼື ແມ່ຄົນດຽວ", "ຢູ່ກັບຍາດພີ່ນ້ອງ", "ຢູ່ຄົນດຽວ"],
    "Family_financial_status": ["ດີ", "ພໍໃຊ້", "ອ່ອນ", "ອ່ອນຫຼາຍ"],
    "Sleep hours night": ["ນ້ອຍກວ່າ 5 ຊົ່ວໂມງ", "5–6 ຊົ່ວໂມງ", "7–8 ຊົ່ວໂມງ", "ຫຼາຍກວ່າ 8 ຊົ່ວໂມງ"],
    "Healthy_meals_freq": ["ທຸກມື້", "3–5 ມື້ຕໍ່ອາທິດ", "1–2 ມື້ຕໍ່ອາທິດ", "ບໍ່ຄ່ອຍ"],
    "Exercise_freq": ["ທຸກມື້", "3–5 ມື້ຕໍ່ອາທິດ", "1–2 ມື້ຕໍ່ອາທິດ", "ບໍ່ເຄີຍ"],
    "Online_time_daily": ["ນ້ອຍກວ່າ 1 ຊົ່ວໂມງ", "1–3 ຊົ່ວໂມງ", "4–6 ຊົ່ວໂມງ", "ຫຼາຍກວ່າ 6 ຊົ່ວໂມງ"],
    "Stress_level_general": ["ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ"],
    "Daytime_tiredness": ["ບໍ່ເຄີຍ", "ບາງຄັ້ງ", "ເລື້ອຍໆ", "ເກືອບທຸກມື້"],
    "Skip_meals_freq": ["ບໍ່ເຄີຍ", "1 ຄັ້ງຕໍ່ອາທິດ", "2–3 ຄັ້ງຕໍ່ອາທິດ", "ເກືອບທຸກມື້"],
    "Time_to_relax": ["ມີ ທຸກມື້", "ມີ ບາງມື້ຕໍ່ອາທິດ", "ບໍ່ຄ່ອຍມີ", "ບໍ່ມີເລີຍ"],
    "Phone_before_sleep": ["ບໍ່ເຄີຍ", "ນ້ອຍກວ່າ 30 ນາທີ", "30–60 ນາທີ", "ຫຼາຍກວ່າ 1 ຊົ່ວໂມງ"],
    "Focus in class": ["ງ່າຍຫຼາຍ", "ງ່າຍ", "ຍາກ", "ຍາກຫຼາຍ"],
    "Worry or tense freq": ["ບໍ່ເຄີຍ", "ບາງຄັ້ງ", "ເລື້ອຍໆ", "ເກືອບທຸກມື້"],
    "Sad or hopeless freq": ["ບໍ່ເຄີຍ", "ບາງຄັ້ງ", "ເລື້ອຍໆ", "ເກືອບທຸກມື້"],
    "Family support when stressed": ["ຫຼາຍ", "ພໍມີ", "ໜ້ອຍ", "ບໍ່ມີ"],
    "Talk to someone when stressed": ["ສະເໝີ", "ບາງຄັ້ງ", "ບໍ່ຄ່ອຍ", "ບໍ່ເຄີຍ"],
    "Academic pressure": ["ຕ່ຳຫຼາຍ", "ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ"],
    "Homework pressure": ["ບໍ່ມີ", "ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ"],
    "Homework pressure3": ["ບໍ່ມີ", "ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ"],
    "CGPA": ["0–5: ຕ່ຳ", "5–7: ປານກາງ", "7–8: ສູງ", "8–10: ສູງຫຼາຍ"],
    "Study satisfaction": ["ຕ່ຳຫຼາຍ", "ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ"],
    "Sleep hours day": ["0–5 ຊົ່ວໂມງ", "6–8 ຊົ່ວໂມງ", "7–9 ຊົ່ວໂມງ"],
    "Study hours day": ["0–3 ຊົ່ວໂມງ", "4–6 ຊົ່ວໂມງ", "7–9 ຊົ່ວໂມງ", "10 ຊົ່ວໂມງຂຶ້ນໄປ"],
    "Depression severity": ["ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ", "ຮຸນແຮງຫຼາຍ"],
    "Anxiety severity": ["ຕ່ຳ", "ປານກາງ", "ສູງ", "ສູງຫຼາຍ", "ຮຸນແຮງຫຼາຍ"],
}

# The same 1–5 scale is used for these mental, stress, support, and well-being items.
LIKERT_1_TO_5_COLS = [
    "Anx_nervous", "Anx_worry_many_things", "Anx_hard_to_relax", "Anx_restless",
    "Anx_upset_easily", "Anx_something_bad", "Anx_sleep_problem", "Aep_sad_most_days",
    "Dep_lose_interest", "Dep_tired_often", "Dep_sleep_problem", "Dep_feel_failure",
    "Dep_cannot_focus", "Dep_slow_thinking", "Dep_hopeless", "Dep_self_harm_thoughts",
    "Stress_schoolwork", "Stress_too_much_manage", "Stress_daily_tasks",
    "Stress_pressure_do_well", "Stress_hard_balance", "Support_someone_to_talk",
    "Support_family", "Support_friends", "Support_understood", "Support_trust_adult",
    "Life_happy", "Self_confident", "Feel_calm", "Handle_problems", "Future_hope"
]
for _col in LIKERT_1_TO_5_COLS:
    ORDINAL_CHOICE_DISPLAY_MAP.setdefault(_col, ["1", "2", "3", "4", "5"])


def get_canonical_column(col):
    col_text = str(col).strip()
    if col_text in LABEL_MAP:
        return col_text

    norm = normalize_col_key(col_text)
    if norm in COLUMN_ALIAS_MAP:
        return COLUMN_ALIAS_MAP[norm]

    for known_col in LABEL_MAP:
        if normalize_col_key(known_col) == norm:
            return known_col

    return col_text


def option_number(value):
    import re
    text = str(value).strip().lower()
    patterns = [
        r"option\s*(\d+)",
        r"ຕົວເລືອກ\s*(\d+)",
        r"^([a-h])\s*[\.|\)]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            if raw.isdigit():
                return int(raw)
            return ord(raw) - ord("a") + 1

    return None


def display_option(value, col=None):
    if value is None:
        return "ບໍ່ລະບຸ"

    value_text = str(value).strip()
    key = normalize_text(value_text)
    canonical_col = get_canonical_column(col) if col else col

    # Fix gender first. Do not let "female" become "male".
    if canonical_col == "Gender":
        if "female" in key or "ຍິງ" in key:
            return "ຍິງ"
        if "male" in key or "ຊາຍ" in key:
            return "ຊາຍ"

    # Use real choice order if the saved value is Option 1, Option 2, etc.
    num = option_number(value_text)
    if canonical_col in ORDINAL_CHOICE_DISPLAY_MAP and num:
        choices = ORDINAL_CHOICE_DISPLAY_MAP[canonical_col]
        if 1 <= num <= len(choices):
            return choices[num - 1]

    # First use exact column-specific choices from the real Google Form.
    if canonical_col in FORM_CHOICE_DISPLAY_MAP:
        form_map = FORM_CHOICE_DISPLAY_MAP[canonical_col]

        for raw_key, shown_text in form_map.items():
            if normalize_text(raw_key) == key:
                return shown_text

        # Then use partial matching. Longer keys go first to avoid wrong matches.
        for raw_key, shown_text in sorted(
            form_map.items(),
            key=lambda item: len(normalize_text(item[0])),
            reverse=True
        ):
            raw_norm = normalize_text(raw_key)
            if raw_norm and raw_norm in key:
                return shown_text

    if key in option_lao_map:
        return option_lao_map[key]

    # If the model has a number as text, keep it simple.
    if value_text.replace(".", "", 1).isdigit():
        return value_text

    # Do not create fake choices. Show the saved text only when no safe match exists.
    return value_text


def slider_help_text(col):
    col_low = str(col).lower()

    if any(k in col_low for k in ["anx", "depress", "dep_", "stress", "hopeless", "self_harm", "worry", "sad"]):
        return "0 = ຍັງບໍ່ໄດ້ຕອບ, 1 = ນ້ອຍຫຼາຍ, 5 = ຫຼາຍຫຼາຍ"

    if any(k in col_low for k in ["support", "happy", "confident", "calm", "handle", "future"]):
        return "0 = ຍັງບໍ່ໄດ້ຕອບ, 1 = ໜ້ອຍຫຼາຍ, 5 = ດີຫຼາຍ"

    return "0 = ຍັງບໍ່ໄດ້ຕອບ, 1 = ຕ່ຳຫຼາຍ, 5 = ສູງຫຼາຍ"


def is_unanswered(value):
    """Return True when a form field has not been answered yet."""
    if value is None:
        return True

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 0.0

    text = str(value).strip().lower()
    return text in ["", "none", "nan", "unknown", "ບໍ່ລະບຸ", "ກະລຸນາເລືອກ"]


def infer_lao_label(col):
    text = normalize_col_key(col)

    if "age" in text:
        return "ອາຍຸ"
    if "gender" in text or "sex" in text:
        return "ເພດ"
    if "province" in text or "district" in text or "village" in text:
        return "ບ່ອນຢູ່"
    if "school level" in text or "grade level" in text or "class" in text:
        return "ລະດັບຊັ້ນຮຽນ"
    if "average grade" in text or "grade" in text or "cgpa" in text or "score" in text:
        return "ຄະແນນການຮຽນ"
    if "academic performance" in text:
        return "ການຮຽນຂອງຕົນ"
    if "living" in text or "live" in text:
        return "ອາໄສຢູ່ກັບໃຜ"
    if "family financial" in text or "financial status" in text:
        return "ຖານະການເງິນຂອງຄອບຄົວ"
    if "sleep" in text:
        return "ການນອນ"
    if "meal" in text or "food" in text:
        return "ການກິນອາຫານ"
    if "exercise" in text:
        return "ການອອກກຳລັງກາຍ"
    if "online" in text or "phone" in text or "screen" in text:
        return "ການໃຊ້ໂທລະສັບ ຫຼື ອອນລາຍ"
    if "focus" in text:
        return "ການຕັ້ງໃຈຮຽນ"
    if "worry" in text or "anx" in text or "tense" in text:
        return "ຄວາມກັງວົນ"
    if "sad" in text or "hopeless" in text or "depress" in text or "dep" in text:
        return "ຄວາມເສົ້າໃນໃຈ"
    if "stress" in text or "pressure" in text:
        return "ຄວາມຄຽດ"
    if "homework" in text:
        return "ວຽກບ້ານ"
    if "study" in text:
        return "ການຮຽນ"
    if "support" in text or "friend" in text or "adult" in text or "talk" in text:
        return "ການຊ່ວຍເຫຼືອ"
    if "happy" in text:
        return "ຄວາມສຸກ"
    if "confident" in text:
        return "ຄວາມໝັ້ນໃຈ"
    if "calm" in text:
        return "ຄວາມສະຫງົບໃຈ"
    if "future" in text or "hope" in text:
        return "ຄວາມຫວັງຕໍ່ອະນາຄົດ"

    return "ຂໍ້ມູນເພີ່ມເຕີມ"


def display_label(col, form_mode="student"):
    canonical_col = get_canonical_column(col)

    if form_mode == "general" and canonical_col in GENERAL_LABEL_MAP:
        return GENERAL_LABEL_MAP[canonical_col]

    if canonical_col in LABEL_MAP:
        return LABEL_MAP[canonical_col]

    # Try to infer the real meaning for any new column.
    return infer_lao_label(col)


def build_section_columns(all_columns, form_mode="student"):
    available = list(all_columns)
    used = set()
    sections = []

    if form_mode == "general":
        section_order = GENERAL_SECTION_ORDER
        section_groups = GENERAL_SECTION_GROUPS
        hidden_columns = GENERAL_HIDDEN_COLUMNS
    else:
        section_order = SECTION_ORDER
        section_groups = SECTION_GROUPS
        hidden_columns = set()

    for section_title in section_order:
        cols = []
        wanted = section_groups[section_title]

        for target_col in wanted:
            for actual_col in available:
                if actual_col in used:
                    continue
                canonical_actual = get_canonical_column(actual_col)
                if canonical_actual in hidden_columns:
                    used.add(actual_col)
                    continue
                if canonical_actual == target_col:
                    cols.append(actual_col)
                    used.add(actual_col)
                    break

        # Put any future unknown columns into the final extra section.
        if section_title == section_order[-1]:
            extra_cols = []
            for c in available:
                canonical_c = get_canonical_column(c)
                if c not in used and canonical_c not in hidden_columns:
                    extra_cols.append(c)
            cols.extend(extra_cols)
            used.update(extra_cols)

        if cols:
            sections.append((section_title, cols))

    return sections



# =========================================================
# DETAILED RESULT ANALYSIS
# =========================================================

POSITIVE_HIGH_COLUMNS = {
    "Average grade", "Academic performance self", "Focus in class", "CGPA",
    "Study satisfaction", "Healthy_meals_freq", "Exercise_freq", "Time_to_relax",
    "Family support when stressed", "Talk to someone when stressed",
    "Support_someone_to_talk", "Support_family", "Support_friends",
    "Support_understood", "Support_trust_adult", "Life_happy",
    "Self_confident", "Feel_calm", "Handle_problems", "Future_hope"
}

PROBLEM_CATEGORY_COLUMNS = {
    "ສຸຂະພາບຈິດ": [
        "Depression severity", "Anxiety severity", "Worry or tense freq",
        "Sad or hopeless freq", "Anx_nervous", "Anx_worry_many_things",
        "Anx_hard_to_relax", "Anx_restless", "Anx_upset_easily",
        "Anx_something_bad", "Aep_sad_most_days", "Dep_lose_interest",
        "Dep_tired_often", "Dep_feel_failure", "Dep_cannot_focus",
        "Dep_slow_thinking", "Dep_hopeless", "Dep_self_harm_thoughts"
    ],
    "ຄວາມຄຽດ": [
        "Stress_level_general", "Academic pressure", "Homework pressure",
        "Homework pressure3", "Stress_schoolwork", "Stress_too_much_manage",
        "Stress_daily_tasks", "Stress_pressure_do_well", "Stress_hard_balance"
    ],
    "ການຮຽນ": [
        "Average grade", "Academic performance self", "Focus in class", "CGPA",
        "Study satisfaction", "Study hours day"
    ],
    "ການນອນ": [
        "Sleep hours night", "Sleep hours day", "Daytime_tiredness",
        "Phone_before_sleep", "Anx_sleep_problem", "Dep_sleep_problem"
    ],
    "ຄອບຄົວ": [
        "Living_with", "Family_financial_status", "Family support when stressed",
        "Support_family"
    ],
    "ການຊ່ວຍເຫຼືອ": [
        "Talk to someone when stressed", "Support_someone_to_talk",
        "Support_friends", "Support_understood", "Support_trust_adult"
    ],
    "ວິຖີຊີວິດ": [
        "Healthy_meals_freq", "Exercise_freq", "Online_time_daily",
        "Skip_meals_freq", "Time_to_relax", "Phone_before_sleep"
    ],
    "ການເງິນ": [
        "Family_financial_status", "Financial stress"
    ],
}

CATEGORY_SIMPLE_ADVICE = {
    "ສຸຂະພາບຈິດ": [
        "ບອກຄູ ພໍ່ແມ່ ຫຼື ຄົນທີ່ໄວ້ໃຈໃຫ້ຮູ້.",
        "ຢ່າຢູ່ຄົນດຽວເມື່ອຮູ້ສຶກໜັກໃຈ.",
        "ພັກຫາຍໃຈຊ້າໆ ແລະ ຂໍຄວາມຊ່ວຍເຫຼືອ."
    ],
    "ຄວາມຄຽດ": [
        "ແບ່ງວຽກໃຫຍ່ເປັນວຽກນ້ອຍ.",
        "ພັກ 5–10 ນາທີຫຼັງຮຽນດົນ.",
        "ຂໍໃຫ້ຄູຊ່ວຍຈັດລຳດັບວຽກ."
    ],
    "ການຮຽນ": [
        "ອ່ານທົບທວນວັນລະ 20–30 ນາທີ.",
        "ຖາມຄູເມື່ອບໍ່ເຂົ້າໃຈ.",
        "ເລືອກບ່ອນຮຽນທີ່ງຽບ ແລະ ຫຼຸດໂທລະສັບ."
    ],
    "ການນອນ": [
        "ພະຍາຍາມນອນໃຫ້ພໍທຸກຄືນ.",
        "ຫຼຸດໂທລະສັບກ່ອນນອນ.",
        "ຈັດເວລານອນ ແລະ ຕື່ນໃຫ້ໃກ້ຄຽງກັນ."
    ],
    "ຄອບຄົວ": [
        "ລອງລົມກັບຄອບຄົວໃນເວລາທີ່ສະຫງົບ.",
        "ບອກຄວາມຮູ້ສຶກດ້ວຍຄຳງ່າຍໆ.",
        "ຖ້າລົມກັນຍາກ ໃຫ້ຂໍຄູ ຫຼື ທີ່ປຶກສາຊ່ວຍ."
    ],
    "ການຊ່ວຍເຫຼືອ": [
        "ເລືອກຄົນ 1 ຄົນທີ່ໄວ້ໃຈແລ້ວລົມດ້ວຍ.",
        "ຢ່າເກັບບັນຫາໄວ້ຄົນດຽວ.",
        "ຂໍເວລາລົມກັບທີ່ປຶກສາໂຮງຮຽນ."
    ],
    "ວິຖີຊີວິດ": [
        "ກິນອາຫານໃຫ້ຄົບມື້.",
        "ຂະຫຍັບຮ່າງກາຍ ຫຼື ຍ່າງເບົາໆ.",
        "ຈຳກັດເວລາອອນລາຍໃຫ້ພໍດີ."
    ],
    "ການເງິນ": [
        "ລົມກັບຜູ້ໃຫຍ່ເມື່ອກັງວົນເລື່ອງເງິນ.",
        "ຢ່າໂທດຕົນເອງເພາະບັນຫານີ້.",
        "ຂໍຄຳແນະນຳຈາກຄູ ຫຼື ທີ່ປຶກສາ."
    ],
}


def _number_from_value(value):
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    except Exception:
        pass

    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        return None


def _text_score(col, value):
    """Convert one answer to a concern score from 0 to 5."""
    canonical_col = get_canonical_column(col)
    shown = display_option(value, canonical_col)
    raw_text = str(value).lower()
    shown_text = str(shown).lower()
    joined = f"{raw_text} {shown_text}"

    if canonical_col in ["Sleep hours night", "Sleep hours day"]:
        if "7" in joined and ("8" in joined or "9" in joined):
            return 0.5
        if "5" in joined and "6" in joined:
            return 2.5
        if "0" in joined and "5" in joined:
            return 4.0
        if "ນ້ອຍ" in joined or "less" in joined:
            return 4.5
        if "ຫຼາຍ" in joined or "more" in joined:
            return 2.5

    if canonical_col == "Online_time_daily":
        if "ນ້ອຍ" in joined or "less" in joined:
            return 0.5
        if "1" in joined and "3" in joined:
            return 1.5
        if "4" in joined and "6" in joined:
            return 3.5
        if "ຫຼາຍ" in joined or "more" in joined:
            return 5.0

    if canonical_col == "Phone_before_sleep":
        if "ບໍ່" in joined or "never" in joined:
            return 0.5
        if "30" in joined and ("ນ້ອຍ" in joined or "less" in joined):
            return 1.5
        if "30" in joined and "60" in joined:
            return 3.0
        if "ຫຼາຍ" in joined or "more" in joined:
            return 5.0

    if canonical_col in ["Healthy_meals_freq", "Exercise_freq", "Time_to_relax"]:
        if "ທຸກ" in joined or "every" in joined or "ມີ ທຸກ" in joined:
            return 0.5
        if "3" in joined and "5" in joined:
            return 1.5
        if "1" in joined and "2" in joined:
            return 3.5
        if "ບໍ່" in joined or "never" in joined or "rare" in joined:
            return 5.0

    if canonical_col in ["Average grade", "CGPA"]:
        if "80" in joined or "8–10" in joined or "8-10" in joined:
            return 0.5
        if "60" in joined or "7–8" in joined or "7-8" in joined:
            return 1.5
        if "40" in joined or "5–7" in joined or "5-7" in joined:
            return 3.0
        if "below" in joined or "ຕ່ຳ" in joined or "0–5" in joined or "0-5" in joined:
            return 5.0

    if canonical_col == "Academic performance self":
        if "ດີຫຼາຍ" in joined or "very good" in joined:
            return 0.5
        if "ດີ" in joined or "good" in joined:
            return 1.5
        if "ປານ" in joined or "average" in joined:
            return 2.5
        if "ຕ່ຳ" in joined or "below" in joined:
            return 4.0
        if "ອ່ອນ" in joined or "poor" in joined:
            return 5.0

    if canonical_col == "Focus in class":
        if "ງ່າຍຫຼາຍ" in joined or "very easy" in joined:
            return 0.5
        if "ງ່າຍ" in joined or "easy" in joined:
            return 1.5
        if "ຍາກຫຼາຍ" in joined or "very hard" in joined:
            return 5.0
        if "ຍາກ" in joined or "hard" in joined:
            return 4.0

    if canonical_col in ["Family_financial_status", "Financial stress"]:
        if "ດີ" in joined or "good" in joined:
            return 0.5
        if "ພໍ" in joined or "fair" in joined:
            return 2.0
        if "ອ່ອນຫຼາຍ" in joined or "very poor" in joined:
            return 5.0
        if "ອ່ອນ" in joined or "poor" in joined:
            return 4.0

    if canonical_col == "Living_with":
        if "ພໍ່ແມ່ທັງສອງ" in joined or "both" in joined:
            return 0.5
        if "ພໍ່" in joined or "ແມ່" in joined or "one parent" in joined:
            return 2.0
        if "ຍາດ" in joined or "relative" in joined:
            return 2.5
        if "ຄົນດຽວ" in joined or "alone" in joined:
            return 4.5

    # General text scale.
    if any(t in joined for t in ["ສູງຫຼາຍ", "ຮຸນແຮງ", "very high", "extreme", "almost every day"]):
        return 5.0
    if any(t in joined for t in ["ສູງ", "ເລື້ອຍ", "high", "often"]):
        return 4.0
    if any(t in joined for t in ["ປານ", "ບາງ", "moderate", "sometimes", "fair"]):
        return 3.0
    if any(t in joined for t in ["ຕ່ຳຫຼາຍ", "very low"]):
        return 1.0
    if any(t in joined for t in ["ຕ່ຳ", "low", "rarely", "ບໍ່ຄ່ອຍ"]):
        return 1.5
    if any(t in joined for t in ["ບໍ່ມີ", "ບໍ່ເຄີຍ", "none", "never", "no"]):
        return 0.5
    if any(t in joined for t in ["ດີຫຼາຍ", "excellent", "very good", "always"]):
        return 0.5 if canonical_col in POSITIVE_HIGH_COLUMNS else 4.5
    if any(t in joined for t in ["ດີ", "good", "yes"]):
        return 1.0 if canonical_col in POSITIVE_HIGH_COLUMNS else 3.5

    return 2.5


CATEGORY_GENERAL_ADVICE = {
    "ສຸຂະພາບຈິດ": [
        "ບອກຄົນໃກ້ຊິດ ຫຼື ຄົນທີ່ໄວ້ໃຈ.",
        "ຢ່າຢູ່ຄົນດຽວເມື່ອຮູ້ສຶກໜັກໃຈ.",
        "ຖ້າຮູ້ສຶກບໍ່ປອດໄພ ໃຫ້ຂໍຊ່ວຍເຫຼືອທັນທີ."
    ],
    "ຄວາມຄຽດ": [
        "ແບ່ງວຽກໃຫຍ່ເປັນວຽກນ້ອຍ.",
        "ພັກ 5–10 ນາທີເມື່ອຄຽດຫຼາຍ.",
        "ຈັດລຳດັບສິ່ງທີ່ຕ້ອງເຮັດກ່ອນ."
    ],
    "ການນອນ": [
        "ພະຍາຍາມນອນໃຫ້ພໍທຸກຄືນ.",
        "ຫຼຸດໂທລະສັບກ່ອນນອນ.",
        "ຈັດເວລານອນ ແລະ ຕື່ນໃຫ້ຄົງທີ່."
    ],
    "ຄອບຄົວ": [
        "ລອງລົມກັບຄອບຄົວໃນເວລາທີ່ສະຫງົບ.",
        "ບອກຄວາມຮູ້ສຶກດ້ວຍຄຳງ່າຍໆ.",
        "ຖ້າລົມກັນຍາກ ໃຫ້ຂໍຄົນກາງທີ່ໄວ້ໃຈຊ່ວຍ."
    ],
    "ການຊ່ວຍເຫຼືອ": [
        "ເລືອກຄົນ 1 ຄົນທີ່ໄວ້ໃຈແລ້ວລົມດ້ວຍ.",
        "ຢ່າເກັບບັນຫາໄວ້ຄົນດຽວ.",
        "ຖ້າຈຳເປັນ ໃຫ້ຂໍພົບຜູ້ໃຫ້ຄຳປຶກສາ."
    ],
    "ວິຖີຊີວິດ": [
        "ກິນອາຫານໃຫ້ຄົບ.",
        "ຂະຫຍັບຮ່າງກາຍເບົາໆທຸກມື້.",
        "ຫຼຸດເວລາອອນລາຍເມື່ອຮູ້ສຶກເມື່ອຍ."
    ],
    "ການເງິນ": [
        "ຈົດລາຍຈ່າຍທີ່ຈຳເປັນກ່ອນ.",
        "ຫຼຸດການໃຊ້ຈ່າຍທີ່ບໍ່ຈຳເປັນ.",
        "ລອງຂໍຄຳແນະນຳຈາກຄົນທີ່ໄວ້ໃຈ."
    ],
}


def advice_for_category(category, form_mode):
    if form_mode == "general":
        return CATEGORY_GENERAL_ADVICE.get(category, CATEGORY_SIMPLE_ADVICE.get(category, []))
    return CATEGORY_SIMPLE_ADVICE.get(category, [])


def concern_score(col, value):
    """Return concern score from 0 to 5. Higher means more concern."""
    if is_unanswered(value):
        return None

    canonical_col = get_canonical_column(col)
    number = _number_from_value(value)

    if number is not None:
        if number <= 0:
            return None
        number = max(1.0, min(5.0, float(number)))
        if canonical_col in POSITIVE_HIGH_COLUMNS:
            return 6.0 - number
        return number

    return _text_score(canonical_col, value)


def column_problem_category(col):
    canonical_col = get_canonical_column(col)
    for category, cols in PROBLEM_CATEGORY_COLUMNS.items():
        if canonical_col in cols:
            return category

    text = normalize_col_key(canonical_col)
    if "stress" in text or "pressure" in text:
        return "ຄວາມຄຽດ"
    if "sleep" in text or "tired" in text:
        return "ການນອນ"
    if "academic" in text or "study" in text or "grade" in text or "focus" in text:
        return "ການຮຽນ"
    if "family" in text or "living" in text:
        return "ຄອບຄົວ"
    if "support" in text or "friend" in text or "adult" in text or "talk" in text:
        return "ການຊ່ວຍເຫຼືອ"
    if "meal" in text or "exercise" in text or "online" in text or "phone" in text:
        return "ວິຖີຊີວິດ"
    if "financial" in text or "money" in text:
        return "ການເງິນ"
    if "anx" in text or "dep" in text or "sad" in text or "hopeless" in text or "worry" in text:
        return "ສຸຂະພາບຈິດ"
    return "ວິຖີຊີວິດ"


def build_detail_analysis(input_dict, risk, class_probabilities, confidence, person_name, form_mode="student"):
    active_problem_categories = [
        category for category in PROBLEM_CATEGORY_COLUMNS
        if not (form_mode == "general" and category == "ການຮຽນ")
    ]
    category_scores = {category: [] for category in active_problem_categories}
    high_items = []

    for col, value in input_dict.items():
        score = concern_score(col, value)
        if score is None:
            continue

        category = column_problem_category(col)
        if form_mode == "general" and category == "ການຮຽນ":
            continue
        category_scores.setdefault(category, []).append(score)

        if score >= 4:
            high_items.append({
                "ຂໍ້ທີ່ຄວນເບິ່ງແຍງ": display_label(col, form_mode),
                "ຄຳຕອບ": display_option(value, col),
                "ລະດັບກັງວົນ": round(score, 1),
                "ກຸ່ມບັນຫາ": category,
            })

    rows = []
    for category, scores in category_scores.items():
        avg_score = float(np.mean(scores)) if scores else 0.0
        percent = round((avg_score / 5.0) * 100.0, 1)
        rows.append({
            "ກຸ່ມບັນຫາ": category,
            "ຄະແນນກັງວົນ (%)": percent,
        })

    concern_df = pd.DataFrame(rows).sort_values(
        "ຄະແນນກັງວົນ (%)", ascending=False
    )

    top_categories = concern_df.head(3)["ກຸ່ມບັນຫາ"].tolist()
    high_items_df = pd.DataFrame(high_items).sort_values(
        "ລະດັບກັງວົນ", ascending=False
    ) if high_items else pd.DataFrame(columns=[
        "ຂໍ້ທີ່ຄວນເບິ່ງແຍງ", "ຄຳຕອບ", "ລະດັບກັງວົນ", "ກຸ່ມບັນຫາ"
    ])

    return {
        "student_name": person_name,
        "form_mode": form_mode,
        "risk": risk,
        "risk_lao": risk_lao_map.get(risk, risk),
        "class_probabilities": class_probabilities,
        "confidence": confidence,
        "concern_df": concern_df,
        "high_items_df": high_items_df,
        "top_categories": top_categories,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def soft_intro(title, subtitle, emoji="🌈"):
    st.markdown(f"""
    <div class='hero-box'>
        <div style='font-size:28px; font-weight:700; color:#23334d;'>{emoji} {title}</div>
        <div style='font-size:16px; color:#42546b; margin-top:8px;'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def soft_card(title, text, emoji="💡"):
    st.markdown(f"""
    <div class='friendly-card'>
        <div class='mini-title'>{emoji} {title}</div>
        <div style='font-size:17px; color:#31445f; line-height:1.6;'>{text}</div>
    </div>
    """, unsafe_allow_html=True)


def soft_badge(text, style_class="risk-low"):
    st.markdown(f"<div class='soft-pill {style_class}'>{text}</div>", unsafe_allow_html=True)


def detail_summary_text(detail):
    risk_lao = detail.get("risk_lao", "ບໍ່ລະບຸ")
    top_categories = detail.get("top_categories", [])
    top_text = ", ".join(top_categories) if top_categories else "ບໍ່ພົບຂໍ້ກັງວົນສູງ"

    return (
        f"ຜົນລວມສະແດງວ່າ ລະດັບຄວາມສ່ຽງແມ່ນ {risk_lao}. "
        f"ກຸ່ມທີ່ຄວນເບິ່ງແຍງກ່ອນແມ່ນ {top_text}. "
        "ຜົນນີ້ໃຊ້ເພື່ອຊ່ວຍຄັດກອງເທົ່ານັ້ນ. "
        "ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
    )


def render_detail_result(detail, form_mode):
    if not detail:
        if form_mode == "student":
            st.info("ຍັງບໍ່ມີຜົນນັກຮຽນ. ກະລຸນາໄປໜ້າທຳນາຍ ແລ້ວເລືອກແບບຟອມນັກຮຽນກ່ອນ.")
        else:
            st.info("ຍັງບໍ່ມີຜົນຄົນທົ່ວໄປ. ກະລຸນາໄປໜ້າທຳນາຍ ແລ້ວເລືອກແບບຟອມບຸກຄົນທົ່ວໄປກ່ອນ.")
        return

    risk_value = detail.get("risk", "Low")
    risk_lao = detail.get("risk_lao", "ບໍ່ລະບຸ")
    risk_class = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "Severe": "risk-severe"
    }.get(risk_value, "risk-low")

    is_general = form_mode == "general"
    result_title = "ຜົນຂອງບຸກຄົນທົ່ວໄປ" if is_general else "ຜົນຂອງນັກຮຽນ"
    name_label = "ຊື່ ຫຼື ລະຫັດຜູ້ຕອບ" if is_general else "ຊື່ ຫຼື ລະຫັດນັກຮຽນ"
    target_note = "ຜົນນີ້ບໍ່ໄດ້ອີງກັບຄຳຖາມໂຮງຮຽນ." if is_general else "ຜົນນີ້ອີງກັບຄຳຕອບຂອງນັກຮຽນ."
    help_people = "ຄອບຄົວ, ໝູ່ສະໜິດ, ຫຼື ຜູ້ໃຫ້ຄຳປຶກສາ" if is_general else "ຄູ, ພໍ່ແມ່, ຫຼື ທີ່ປຶກສາ"

    top_categories = detail.get("top_categories", [])
    top_text = ", ".join(top_categories[:3]) if top_categories else "ບໍ່ພົບຈຸດທີ່ກັງວົນຫຼາຍ"

    st.markdown(f"""
    <div class='hero-box'>
        <div style='font-size:22px; font-weight:700; color:#31445f;'>🌈 {result_title}</div>
        <div style='font-size:16px; color:#42546b; margin-top:8px;'>{target_note}</div>
        <div class='soft-pill {risk_class}'>ລະດັບຄວາມສ່ຽງ: {risk_lao}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='friendly-card'>
            <div class='mini-title'>👤 {name_label}</div>
            <div class='big-number' style='font-size:24px'>{detail.get('student_name', '-')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='friendly-card'>
            <div class='mini-title'>🎯 ຄວາມໝັ້ນໃຈຂອງໂມເດວ</div>
            <div class='big-number'>{detail.get('confidence', 0):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='friendly-card'>
            <div class='mini-title'>🕒 ເວລາອັບເດດ</div>
            <div style='font-size:20px; font-weight:700; color:#22304a;'>{detail.get('timestamp', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='friendly-card'>", unsafe_allow_html=True)
    st.subheader("ສະຫຼຸບແບບສັ້ນ")
    if risk_value == 'Low':
        st.success("ຜົນຢູ່ໃນລະດັບຕ່ຳ. ຍັງຄວນດູແລການນອນ, ອາຫານ, ອາລົມ ແລະ ຄົນຮອບຂ້າງ.")
    elif risk_value == 'Moderate':
        st.warning(f"ຜົນຢູ່ໃນລະດັບປານກາງ. ຄວນເລີ່ມດູແລຕົນເອງຫຼາຍຂຶ້ນ ແລະ ລອງຄຸຍກັບ {help_people}.")
    else:
        st.error(f"ຜົນຢູ່ໃນລະດັບສູງ. ຄວນຮີບຄຸຍກັບ {help_people} ເພື່ອຂໍການຊ່ວຍເຫຼືອ.")

    st.write(f"**ຈຸດທີ່ຄວນເບິ່ງແຍງກ່ອນ:** {top_text}")
    st.markdown("</div>", unsafe_allow_html=True)

    concern_df = detail.get("concern_df", pd.DataFrame())
    if not concern_df.empty:
        st.markdown("<div class='friendly-card'>", unsafe_allow_html=True)
        st.subheader("ກຸ່ມບັນຫາທີ່ຄວນໃສ່ໃຈ")
        st.caption("ແຖບທີ່ສູງກວ່າ ໝາຍຄວາມວ່າ ກຸ່ມນັ້ນຄວນເບິ່ງແຍງຫຼາຍກວ່າ")
        st.bar_chart(data=concern_df, x="ກຸ່ມບັນຫາ", y="ຄະແນນກັງວົນ (%)")
        st.markdown("</div>", unsafe_allow_html=True)

    high_items_df = detail.get("high_items_df", pd.DataFrame())
    st.markdown("<div class='friendly-card'>", unsafe_allow_html=True)
    st.subheader("ສິ່ງທີ່ຄວນລະວັງກ່ອນ")
    if high_items_df.empty:
        st.success("ບໍ່ພົບຂໍ້ທີ່ນ່າກັງວົນຫຼາຍ. ຂໍໃຫ້ຮັກສາການດູແລຕົນເອງໄວ້.")
    else:
        for _, row in high_items_df.head(5).iterrows():
            st.markdown(f"""
            <div class='tip-box'>
                <b>• {row['ຂໍ້ທີ່ຄວນເບິ່ງແຍງ']}</b><br>
                ຄຳຕອບ: {row['ຄຳຕອບ']}<br>
                ຢູ່ໃນກຸ່ມ: {row['ກຸ່ມບັນຫາ']}
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='friendly-card'>", unsafe_allow_html=True)
    st.subheader("ວິທີດູແລຕົນເອງເບື້ອງຕົ້ນ")
    if not top_categories:
        st.write("• ນອນໃຫ້ພໍ")
        st.write("• ກິນອາຫານໃຫ້ຄົບ")
        st.write("• ຄຸຍກັບຄົນທີ່ໄວ້ໃຈເມື່ອບໍ່ສະບາຍໃຈ")
    else:
        for category in top_categories:
            st.markdown(f"##### {category}")
            for advice in advice_for_category(category, form_mode):
                st.write(f"• {advice}")
    st.markdown("</div>", unsafe_allow_html=True)

    if detail.get("risk") == "Severe":
        st.markdown(f"""
        <div class='danger-box'>
            <b>ສຳຄັນຫຼາຍ</b><br>
            ຖ້າຮູ້ສຶກຢາກທຳຮ້າຍຕົນເອງ ຫຼື ຮູ້ສຶກບໍ່ປອດໄພ,
            ໃຫ້ບອກ {help_people} ທັນທີ. ຢ່າຢູ່ຄົນດຽວ.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"ຖ້າມື້ໃດຮູ້ສຶກບໍ່ສະບາຍໃຈຫຼາຍຂຶ້ນ ໃຫ້ຮີບບອກ {help_people}.")

    st.caption("ໜ້ານີ້ໃຊ້ເພື່ອຊ່ວຍຄັດກອງເທົ່ານັ້ນ. ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ.")


def show_detail_result_page():
    soft_intro(
        "ລາຍງານຜົນລະອຽດ",
        "ຫຼັງຈາກກົດທຳນາຍ ລະບົບຈະພາມາໜ້ານີ້ເອງ. ເລືອກປະເພດລາຍງານໄດ້ທີ່ນີ້.",
        "📋"
    )

    latest_mode = st.session_state.get("detail_mode_to_show", "student")
    report_options = ["🎒 ລາຍງານນັກຮຽນ", "👥 ລາຍງານຄົນທົ່ວໄປ"]
    default_index = 1 if latest_mode == "general" else 0

    selected_report = st.radio(
        "ເລືອກລາຍງານ",
        report_options,
        index=default_index,
        horizontal=True,
        key=f"detail_report_mode_{latest_mode}"
    )

    st.divider()

    if selected_report.startswith("🎒"):
        render_detail_result(
            st.session_state.get("latest_student_detail_result"),
            "student"
        )
    else:
        render_detail_result(
            st.session_state.get("latest_general_detail_result"),
            "general"
        )

# =========================================================
# SIDEBAR
# =========================================================

PAGES = [
    "ໜ້າຫຼັກ",
    "ຜົນຂອງໂມເດວ",
    "ທຳນາຍຄວາມສ່ຽງ",
    "ລາຍງານຜົນລະອຽດ",
    "ກ່ຽວກັບ"
]

st.sidebar.title("🏠 ເມນູ")
st.sidebar.caption(APP_VERSION)
st.sidebar.info(
    "ເລືອກໜ້າຢູ່ທາງຂ້າງນີ້. "
    "ໃນໂທລະສັບ ກົດປຸ່ມ >> ເພື່ອເປີດເມນູ."
)

if st.session_state.get("mobile_page_selector") == "ຜົນລະອຽດ":
    st.session_state["mobile_page_selector"] = "ລາຍງານຜົນລະອຽດ"

if "forced_page" not in st.session_state:
    st.session_state["forced_page"] = None


def clear_forced_page():
    st.session_state["forced_page"] = None


selected_page = st.sidebar.selectbox(
    "ເລືອກໜ້າ",
    PAGES,
    index=0,
    key="mobile_page_selector",
    on_change=clear_forced_page
)

page = st.session_state.get("forced_page") or selected_page


# =========================================================
# HOME PAGE
# =========================================================

if page == "ໜ້າຫຼັກ":
    soft_intro(
        "ຍິນດີຕ້ອນຮັບ",
        "ແອັບນີ້ຊ່ວຍຄັດກອງຄວາມຮູ້ສຶກເບື້ອງຕົ້ນ. ອ່ານງ່າຍ, ສີນຸ່ມ, ແລະ ເຫັນຜົນໄດ້ຊັດ.",
        "🏡"
    )

    st.info(
        "ແອັບນີ້ຊ່ວຍຄັດກອງຄວາມສ່ຽງເບື້ອງຕົ້ນ. "
        "ຖ້າຜົນບອກວ່າຄວນຕິດຕາມ, ລະບົບຈະຊ່ວຍແຈ້ງທີ່ປຶກສາ."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("ຄວາມແມ່ນຍຳ", "92.81%")
    col2.metric("ຄວາມແມ່ນຍຳແບບສົມດຸນ", "96.99%")
    col3.metric("ຄະແນນລວມ", "96.28%")

    soft_card("ໂມເດວພ້ອມແລ້ວ", "ຕອນນີ້ລະບົບພ້ອມໃຫ້ເຈົ້າປ້ອນຂໍ້ມູນ ແລະ ເບິ່ງຜົນໄດ້ເລີຍ.", "✅")

    model_data = pd.DataFrame({
        "ຕົວຊີ້ວັດ": [
            "ຄວາມແມ່ນຍຳໃນຊຸດທົດສອບ",
            "ຄວາມແມ່ນຍຳແບບສົມດຸນ",
            "ຄະແນນລວມ"
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
    soft_intro(
        "ຜົນຂອງໂມເດວ",
        "ໜ້ານີ້ບອກວ່າ ລະບົບທຳວຽກໄດ້ດີປານໃດ ໂດຍໃຊ້ຄຳທີ່ອ່ານງ່າຍ.",
        "📊"
    )

    soft_card("ອ່ານງ່າຍ", "ຕົວເລກຫນ້ານີ້ແມ່ນຜົນທົດສອບຂອງລະບົບກັບຂໍ້ມູນຈິງ.", "💙")

    st.subheader("ຕາຕະລາງຜົນການຈັດປະເພດ")

    results_data = {
        "ລະດັບຄວາມສ່ຽງ": ["ຕ່ຳ", "ປານກາງ", "ສູງ"],
        "ຄວາມແມ່ນຍຳເມື່ອທຳນາຍ": [0.95, 0.94, 0.97],
        "ການກວດພົບໄດ້ຖືກ": [0.96, 0.93, 0.98],
        "ຄະແນນລວມ": [0.95, 0.94, 0.97]
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
    soft_intro(
        "ກອກຂໍ້ມູນເພື່ອຄັດກອງ",
        "ຕອບຕາມຄວາມຈິງໄດ້ເລີຍ. ບໍ່ມີຄຳຕອບຖືກ ຫຼື ຜິດ.",
        "📝"
    )

    st.info(
        "ເລກ 0 ໝາຍເຖິງ ຍັງບໍ່ໄດ້ຕອບ. ເລກ 1 ແມ່ນນ້ອຍຫຼາຍ. ເລກ 5 ແມ່ນຫຼາຍຫຼາຍ. "
        "ກະລຸນາຕອບໃຫ້ຄົບທຸກຂໍ້ກ່ອນກົດສົ່ງ."
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

    def choose_neutral_category(col):
        options = cat_options.get(col, [])
        clean_options = []
        for option in options:
            option_text = str(option).strip()
            if normalize_text(option_text) not in ["nan", "none", ""]:
                clean_options.append(option)

        if not clean_options:
            return "Unknown"

        preferred_words = [
            "moderate", "average", "fair", "sometimes", "some",
            "good", "m.4", "m4", "ມ.4", "60", "5-7", "5–7"
        ]
        for word in preferred_words:
            for option in clean_options:
                option_text = normalize_text(option)
                if word in option_text:
                    return option

        return clean_options[0]

    def complete_input_for_model(input_dict):
        complete = dict(input_dict)
        for col in common_cols:
            if col in complete and not is_unanswered(complete[col]):
                continue
            if col in num_cols:
                complete[col] = 3.0
            else:
                complete[col] = choose_neutral_category(col)
        return complete

    st.markdown(
        """
        <div class='friendly-card' style='padding:14px 16px;'>
            <div style='font-size:18px;font-weight:800;color:#22304a;'>🌟 ເລືອກແບບຟອມກ່ອນ</div>
            <div style='font-size:15px;color:#516178;margin-top:4px;'>ກົດເລືອກວ່າຜູ້ຕອບແມ່ນນັກຮຽນ ຫຼື ບຸກຄົນທົ່ວໄປ</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    form_choice = st.radio(
        "ເລືອກປະເພດແບບຟອມ",
        ["ນັກຮຽນ", "ບຸກຄົນທົ່ວໄປ"],
        horizontal=True,
        key="form_mode_choice"
    )

    form_mode = "general" if form_choice == "ບຸກຄົນທົ່ວໄປ" else "student"

    if form_mode == "general":
        st.info(
            "ໂໝດນີ້ໃຊ້ສຳລັບຄົນທົ່ວໄປທຸກໄວ. "
            "ຄຳຖາມທີ່ເກື່ອນກັບໂຮງຮຽນຈະຖືກເຊື່ອງ. "
            "ລະບົບຈະໃສ່ຄ່າກາງໃຫ້ອັດຕະໂນມັດ. "
            "ຜົນນີ້ເປັນການຄັດກອງເບື້ອງຕົ້ນເທົ່ານັ້ນ."
        )
    else:
        st.info("ໂໝດນີ້ເໝາະກັບນັກຮຽນ ຫຼື ໄວຮຽນ.")

    section_columns = build_section_columns(common_cols, form_mode)

    def input_field(col):
        label = display_label(col, form_mode)
        safe_key = f"input_{col}"

        if col in num_cols:
            return float(
                st.slider(
                    label,
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    help=slider_help_text(col),
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
                label_text = display_option(option, col)

                if label_text in display_to_original:
                    label_text = f"{label_text} {i + 1}"

                display_labels.append(label_text)
                display_to_original[label_text] = option

            selected_label = st.selectbox(
                label,
                options=display_labels,
                index=None,
                placeholder="ກະລຸນາເລືອກ",
                key=safe_key
            )

            if selected_label is None:
                return None

            return display_to_original[selected_label]

        text_value = st.text_input(
            label,
            value="",
            placeholder="ບໍ່ລະບຸ",
            key=safe_key
        )

        return text_value if text_value else None

    def show_group(title, cols):
        st.markdown(f"### {title}")

        with st.container(border=True):
            if len(cols) == 0:
                st.caption("ບໍ່ພົບຊ່ອງຂໍ້ມູນ")
            else:
                for col in cols:
                    user_input[col] = input_field(col)
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    def format_form_answers_lao(input_dict):
        lines = []

        for key, value in input_dict.items():
            lao_key = display_label(key, form_mode)
            lao_value = display_option(value, key)
            lines.append(f"{lao_key}: {lao_value}")

        return "\n".join(lines)

    st.subheader("ປ້ອນຂໍ້ມູນນັກຮຽນ" if form_mode == "student" else "ປ້ອນຂໍ້ມູນຄົນທົ່ວໄປ")

    for section_title, section_cols in section_columns:
        show_group(section_title, section_cols)

    st.divider()

    st.subheader("ຂໍ້ມູນສຳລັບສົ່ງແຈ້ງເຕືອນ")

    person_name_label = "ຊື່ນັກຮຽນ ຫຼື ລະຫັດນັກຮຽນ" if form_mode == "student" else "ຊື່ ຫຼື ລະຫັດຜູ້ຕອບ"

    student_name_for_report = st.text_input(
        person_name_label,
        key="student_name_auto"
    )

    consent_alert = st.checkbox(
        "ຂ້ອຍຍິນຍອມໃຫ້ສົ່ງຜົນຄັດກອງນີ້ໄປຫາທີ່ປຶກສາ",
        key="consent_auto"
    )

    st.divider()

    unanswered_cols_now = [
        col for col, value in user_input.items()
        if is_unanswered(value)
    ]

    missing_submit_info_now = []
    if is_unanswered(student_name_for_report):
        missing_submit_info_now.append(person_name_label)
    if not consent_alert:
        missing_submit_info_now.append("ການຍິນຍອມໃຫ້ສົ່ງຜົນໄປຫາທີ່ປຶກສາ")

    form_ready_to_submit = (
        len(unanswered_cols_now) == 0
        and len(missing_submit_info_now) == 0
    )

    if not form_ready_to_submit:
        st.warning("ກະລຸນາຕອບຂໍ້ມູນໃຫ້ຄົບກ່ອນກົດສົ່ງ.")
        with st.expander("ເບິ່ງຂໍ້ທີ່ຍັງຂາດ"):
            missing_labels = [
                display_label(col, form_mode) for col in unanswered_cols_now
            ] + missing_submit_info_now

            if missing_labels:
                st.write(" • " + "\n • ".join(missing_labels[:30]))
                if len(missing_labels) > 30:
                    st.caption(f"ຍັງມີອີກ {len(missing_labels) - 30} ຂໍ້")

    submit_clicked = st.button(
        "ທຳນາຍຄວາມສ່ຽງ",
        use_container_width=True,
        disabled=not form_ready_to_submit
    )

    if submit_clicked:
        unanswered_cols = [
            col for col, value in user_input.items()
            if is_unanswered(value)
        ]

        missing_submit_info = []
        if is_unanswered(student_name_for_report):
            missing_submit_info.append(person_name_label)
        if not consent_alert:
            missing_submit_info.append("ການຍິນຍອມໃຫ້ສົ່ງຜົນໄປຫາທີ່ປຶກສາ")

        if unanswered_cols or missing_submit_info:
            st.warning("ກະລຸນາຕອບທຸກຂໍ້ກ່ອນກົດສົ່ງ.")
            missing_labels = [
                display_label(col, form_mode) for col in unanswered_cols
            ] + missing_submit_info
            st.write(" • " + "\n • ".join(missing_labels[:30]))
            if len(missing_labels) > 30:
                st.caption(f"ຍັງມີອີກ {len(missing_labels) - 30} ຂໍ້")
            st.stop()

        model_input = complete_input_for_model(user_input) if form_mode == "general" else dict(user_input)

        input_df = pd.DataFrame([model_input])
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

        conclusion = make_conclusion(risk, form_mode)

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

        detail_result = build_detail_analysis(
            user_input,
            risk,
            class_probabilities,
            confidence,
            student_name_for_report,
            form_mode
        )
        if form_mode == "student":
            st.session_state["latest_student_detail_result"] = detail_result
            result_tab_name = "ຜົນນັກຮຽນ"
        else:
            st.session_state["latest_general_detail_result"] = detail_result
            result_tab_name = "ຜົນຄົນທົ່ວໄປ"
        st.session_state["latest_detail_result"] = detail_result

        st.info(
            "ລາຍງານຜົນລະອຽດຖືກກຽມໄວ້ແລ້ວ. "
            "ກຳລັງເປີດໜ້າ ລາຍງານຜົນລະອຽດ ໃຫ້ອັດຕະໂນມັດ."
        )

        st.caption(
            "ຜົນນີ້ເປັນການຄັດກອງເບື້ອງຕົ້ນ. "
            "ຖ້າຮູ້ສຶກບໍ່ດີ ຫຼື ບໍ່ປອດໄພ, ໃຫ້ບອກຄົນທີ່ໄວ້ໃຈທັນທີ."
        )

        send_to_counselor = risk in AUTO_COUNSELOR_LEVELS

        if send_to_counselor:

            if not student_name_for_report:
                st.warning(f"ກະລຸນາໃສ່{person_name_label}ກ່ອນສົ່ງ")

            elif not consent_alert:
                st.warning("ກະລຸນາກົດຍິນຍອມກ່ອນສົ່ງການແຈ້ງເຕືອນ")

            else:
                report_data = {
                    "student_name": student_name_for_report,
                    "form_type": form_choice,
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
                    "alert_flow": "ສົ່ງໃຫ້ທີ່ປຶກສາເທົ່ານັ້ນ",
                    "disclaimer": "ຜົນນີ້ແມ່ນການຄັດກອງ ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ."
                }

                sent = send_report_to_n8n(report_data)

                if sent:
                    st.success(
                        "ສົ່ງຜົນໄປຫາລະບົບແຈ້ງເຕືອນສຳເລັດແລ້ວ. "
                        "ທີ່ປຶກສາຈະໄດ້ຮັບການແຈ້ງເຕືອນ."
                    )
                else:
                    st.error("ສົ່ງຜົນໄປຫາລະບົບແຈ້ງເຕືອນບໍ່ສຳເລັດ")

        else:
            st.info(
                "ຜົນຄວາມສ່ຽງຕ່ຳ. "
                "ບໍ່ມີການສົ່ງອີເມວອັດຕະໂນມັດ."
            )

        st.session_state["detail_mode_to_show"] = form_mode
        st.session_state["forced_page"] = "ລາຍງານຜົນລະອຽດ"
        st.success("ກຳລັງເປີດໜ້າ ລາຍງານຜົນລະອຽດ ໃຫ້ເຈົ້າ...")
        st.rerun()


# =========================================================
# DETAILED RESULT PAGE
# =========================================================

elif page in ["ລາຍງານຜົນລະອຽດ", "ຜົນລະອຽດ"]:
    show_detail_result_page()


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ກ່ຽວກັບ":
    st.header("ກ່ຽວກັບແອັບນີ້")

    st.write(
        """
        ແອັບນີ້ໃຊ້ໂມເດວປັນຍາປະດິດທີ່ຝຶກແລ້ວ
        ເພື່ອຊ່ວຍຄັດກອງຄວາມສ່ຽງເບື້ອງຕົ້ນ.

        ແອັບນີ້ມີ 2 ແບບຟອມ:
        1. ນັກຮຽນ
        2. ຄົນທົ່ວໄປ

        ຜົນຈະສະແດງເປັນ 3 ລະດັບ:
        ຕ່ຳ, ປານກາງ, ແລະ ສູງ.

        ຜົນນີ້ເປັນການຄັດກອງເບື້ອງຕົ້ນເທົ່ານັ້ນ.
        ບໍ່ແມ່ນການວິນິດໄສທາງການແພດ.
        """
    )
