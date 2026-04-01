import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Resume Generator", layout="centered")

st.title("🚀 AI Resume Generator (LayoutGAN)")
st.markdown("Create a professional resume using AI")

# ---------------- USER INPUT ----------------
name = st.text_input("Enter your name")
summary = st.text_area("Enter summary")
skills = st.text_area("Enter skills (comma separated)")
education = st.text_area("Enter education")
experience = st.text_area("Enter experience")

photo = st.file_uploader("Upload Profile Photo", type=["png", "jpg", "jpeg"])

generate = st.button("Generate Resume")

# ---------------- MODEL ----------------
class LayoutGAN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
        super(LayoutGAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cpu")
model = LayoutGAN().to(device)
model.load_state_dict(torch.load("layoutgan_resume.pth", map_location=device))
model.eval()

# ---------------- LAYOUT FUNCTION ----------------
def generate_layout():
    sections = {
        "Summary": [0.1, 0.1, 0.8, 0.2],
        "Skills": [0.1, 0.35, 0.8, 0.1],
        "Education": [0.1, 0.5, 0.8, 0.1],
        "Experience": [0.1, 0.65, 0.8, 0.2]
    }

    input_tensor = torch.tensor(list(sections.values()), dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor).numpy()

    return dict(zip(sections.keys(), output))

# ---------------- SHOW LAYOUT ----------------
def show_layout(layout):
    fig, ax = plt.subplots()

    for section, (x, y, w, h) in layout.items():
        rect = plt.Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(x, y, section)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    st.pyplot(fig)

# ---------------- PDF FUNCTION ----------------
def create_pdf(name, summary, skills, education, experience, photo):
    doc = SimpleDocTemplate("resume.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Name
    elements.append(Paragraph(f"<b>{name}</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Photo
    if photo is not None:
        with open("temp.jpg", "wb") as f:
            f.write(photo.read())
        elements.append(Image("temp.jpg", width=100, height=100))
        elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    elements.append(Paragraph(summary, styles["BodyText"]))
    elements.append(Spacer(1, 12))

    # Skills (bullets)
    elements.append(Paragraph("<b>Skills</b>", styles["Heading2"]))
    skills_list = skills.split(",")
    for skill in skills_list:
        elements.append(Paragraph(f"• {skill.strip()}", styles["BodyText"]))
    elements.append(Spacer(1, 12))

    # Education
    elements.append(Paragraph("<b>Education</b>", styles["Heading2"]))
    elements.append(Paragraph(education, styles["BodyText"]))
    elements.append(Spacer(1, 12))

    # Experience
    elements.append(Paragraph("<b>Experience</b>", styles["Heading2"]))
    elements.append(Paragraph(experience, styles["BodyText"]))

    doc.build(elements)

# ---------------- MAIN LOGIC ----------------
if generate:
    layout = generate_layout()

    st.subheader("📐 Generated Layout")
    st.write(layout)

    show_layout(layout)

    create_pdf(name, summary, skills, education, experience, photo)

    st.success("✅ Resume Generated Successfully!")

    with open("resume.pdf", "rb") as f:
        st.download_button("📄 Download Resume", f, file_name="resume.pdf")