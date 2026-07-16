import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import time

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="RiceVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: 'Segoe UI';
}

.main{
    background-color:#f7fbf7;
}

.hero{
background:linear-gradient(90deg,#2E7D32,#66BB6A);
padding:30px;
border-radius:18px;
color:white;
text-align:center;
margin-bottom:20px;
box-shadow:0px 5px 15px rgba(0,0,0,0.15);
}

.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0 6px 15px rgba(0,0,0,0.12);
margin-top:15px;

}

.metric{

background:#ffffff;
padding:18px;
border-radius:12px;
text-align:center;
box-shadow:0 4px 12px rgba(0,0,0,0.1);

}

.footer{

text-align:center;
color:gray;
margin-top:50px;

}

.stButton>button{

background:#2E7D32;
color:white;
border-radius:12px;
height:50px;
width:100%;
font-size:18px;
font-weight:bold;
border:none;

}

.stButton>button:hover{

background:#1B5E20;

}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909766.png",
    width=120
)

st.sidebar.title("🌾 RiceVision AI")

st.sidebar.markdown("---")

st.sidebar.info(
"""
RiceVision AI uses a CNN model trained on five rice varieties.

Supported Classes

• Arborio

• Basmati

• Ipsala

• Jasmine

• Karacadag
"""
)

st.sidebar.markdown("---")

st.sidebar.success("TensorFlow CNN Model")

# ------------------------------
# Hero Section
# ------------------------------

st.markdown("""

<div class='hero'>

<h1>🌾 RiceVision AI</h1>

<h4>Deep Learning Based Rice Variety Classification</h4>

<p>
Upload a clear image of a single rice grain to identify its variety.
</p>

</div>

""", unsafe_allow_html=True)

# ------------------------------
# Load Model
# ------------------------------

@st.cache_resource

def load_model():

    return tf.keras.models.load_model("../model.keras")

model = load_model()

classes = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
]

rice_info = {

"Arborio":{

"country":"Italy",

"use":"Risotto",

"description":"Short grain creamy rice ideal for Italian dishes."

},

"Basmati":{

"country":"India & Pakistan",

"use":"Biryani and Pulao",

"description":"Long aromatic rice with excellent fragrance."

},

"Ipsala":{

"country":"Turkey",

"use":"Traditional Turkish dishes",

"description":"Medium grain rice cultivated in Turkey."

},

"Jasmine":{

"country":"Thailand",

"use":"Asian Cuisine",

"description":"Fragrant rice with soft texture."

},

"Karacadag":{

"country":"Turkey",

"use":"Traditional Meals",

"description":"Ancient rice variety grown near Mount Karacadag."

}

}
# ------------------------------
# Upload Section
# ------------------------------

st.markdown("## 📤 Upload Rice Image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown("### 📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with st.spinner("Analyzing image using CNN..."):

        time.sleep(1)

        img = image.resize((128,128))

        img = np.array(img)/255.0

        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img,verbose=0)

        index = np.argmax(prediction)

        confidence = prediction[0][index]*100

        rice = classes[index]

    with col2:

        st.markdown("### 🎯 Prediction")

        st.success(rice)

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

        st.progress(float(confidence/100))

        if confidence>=95:

            st.success("Excellent prediction confidence")

        elif confidence>=80:

            st.info("Good prediction confidence")

        else:

            st.warning(
                "Confidence is relatively low. Try a clearer image."
            )

    st.markdown("---")

    st.markdown("## 🌾 Rice Information")

    c1,c2,c3=st.columns(3)

    with c1:

        st.markdown(f"""

        <div class='metric'>

        <h3>🌍 Origin</h3>

        <h2>{rice_info[rice]['country']}</h2>

        </div>

        """,unsafe_allow_html=True)

    with c2:

        st.markdown(f"""

        <div class='metric'>

        <h3>🍚 Common Use</h3>

        <h2>{rice_info[rice]['use']}</h2>

        </div>

        """,unsafe_allow_html=True)

    with c3:

        st.markdown(f"""

        <div class='metric'>

        <h3>🤖 Predicted Class</h3>

        <h2>{rice}</h2>

        </div>

        """,unsafe_allow_html=True)

    st.markdown("### 📖 Description")

    st.info(

        rice_info[rice]["description"]

    )

    st.markdown("---")

    st.markdown("## 📊 Prediction Probabilities")

    probability = pd.DataFrame({

        "Rice Variety":classes,

        "Probability":prediction[0]*100

    })

    fig = px.bar(

        probability,

        x="Rice Variety",

        y="Probability",

        text="Probability",

        color="Probability",

        color_continuous_scale="Greens"

    )

    fig.update_traces(

        texttemplate="%{text:.2f}%",
        textposition="outside"

    )

    fig.update_layout(

        height=450,

        showlegend=False

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("### 📈 Confidence for Each Class")

    for cls,prob in zip(classes,prediction[0]):

        st.write(f"**{cls}**")

        st.progress(float(prob))

    # ------------------------------
    # Model Insights
    # ------------------------------

    st.markdown("---")
    st.markdown("## 📊 Model Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card'>
        <h3>🧠 Model</h3>
        <h2>CNN</h2>
        <p>Convolutional Neural Network</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <h3>📷 Image Size</h3>
        <h2>128 × 128</h2>
        <p>RGB Input</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='card'>
        <h3>🏷 Classes</h3>
        <h2>{len(classes)}</h2>
        <p>Rice Varieties</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------
    # Expandable Information
    # ------------------------------

    st.markdown("---")

    with st.expander("🌾 Learn about this Rice Variety"):

        st.write("### Predicted Variety")
        st.write(rice)

        st.write("### Country")
        st.write(rice_info[rice]["country"])

        st.write("### Common Uses")
        st.write(rice_info[rice]["use"])

        st.write("### Description")
        st.write(rice_info[rice]["description"])

    # ------------------------------
    # Tips
    # ------------------------------

    st.markdown("---")

    st.markdown("## 💡 Tips for Better Prediction")

    st.info("""
✅ Upload **a single rice grain** instead of a handful.

✅ Use a plain white or black background.

✅ Ensure good lighting.

✅ Avoid blurry or low-resolution images.

✅ Crop the rice grain so it occupies most of the image.
""")

    # ------------------------------
    # About Model
    # ------------------------------

    st.markdown("---")

    st.markdown("## 🤖 About the Model")

    st.write("""
This application uses a **Convolutional Neural Network (CNN)** trained on
five different rice varieties.

The uploaded image is resized to **128×128 pixels**, normalized, and then
passed through the trained deep learning model to predict the rice variety.
""")

    st.markdown(f"""
    <div style="
    background:linear-gradient(90deg,#43A047,#66BB6A);
    padding:25px;
    border-radius:18px;
    text-align:center;
    color:white;
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
    ">

    <h1>🌾 {rice}</h1>

    <h4>Predicted Rice Variety</h4>

    </div>

    """,unsafe_allow_html=True)

    fig = px.pie(

    values=[confidence,100-confidence],

    names=["Confidence",""],

    hole=0.75,

    color_discrete_sequence=["green","#ECECEC"]

    )

    fig.update_layout(

    height=350,

    showlegend=False,

    annotations=[

    dict(

    text=f"<b>{confidence:.1f}%</b>",

    x=0.5,

    y=0.5,

    font_size=28,

    showarrow=False

    )

    ]

    )

    st.plotly_chart(fig,use_container_width=True)

    report=f"""

    RiceVision AI Report

    Prediction : {rice}

    Confidence : {confidence:.2f}%

    Country : {rice_info[rice]['country']}

    Use : {rice_info[rice]['use']}

    """

    st.download_button(

    "📄 Download Report",

    report,

    file_name="prediction_report.txt"

    )

    st.sidebar.metric(

    "Classes",

    5

    )

    st.sidebar.metric(

    "Model",

    "CNN"

    )

    st.sidebar.metric(

    "Accuracy",

    "99.5%"

    )