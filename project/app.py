import streamlit as st
import sqlite3
import os
import uuid
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
from denoising_model import simple_denoise
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Setup folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("clean", exist_ok=True)

# Database setup
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

# Tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    email TEXT,
    phone TEXT,
    gender TEXT,
    address TEXT
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    filename TEXT,
    upload_time TEXT
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS clean_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER,
    clean_filename TEXT,
    psnr REAL,
    ssim REAL
)''')

conn.commit()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Functions
def register_user(username, password, email, phone, gender, address):
    try:
        cursor.execute("INSERT INTO users (username, password, email, phone, gender, address) VALUES (?, ?, ?, ?, ?, ?)",
                       (username, password, email, phone, gender, address))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

def save_upload(user_id, filename):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO uploads (user_id, filename, upload_time) VALUES (?, ?, ?)",
                   (user_id, filename, now))
    conn.commit()
    return cursor.lastrowid

def save_clean_image(upload_id, clean_filename, psnr_score, ssim_score):
    cursor.execute("INSERT INTO clean_images (upload_id, clean_filename, psnr, ssim) VALUES (?, ?, ?, ?)",
                   (upload_id, clean_filename, psnr_score, ssim_score))
    conn.commit()

# UI
st.title(" Astronomical Image Noise Reduction with Machine Learning Techniques")

menu = ["Register", "Login", "Upload & Denoise", "More Upload & Denoise"]
choice = st.sidebar.selectbox("Navigation", menu)

# Register
if choice == "Register":
    st.subheader("Register New User")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    address = st.text_area("Address")
    if st.button("Register"):
        if register_user(username, password, email, phone, gender, address):
            st.success("Registered Successfully. You can now login.")
        else:
            st.error("Username already exists.")

# Login
elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.success(f"Welcome, {username}!")
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
        else:
            st.error("Invalid credentials.")

elif choice == "More Upload & Denoise":
    if not st.session_state.logged_in:
        st.warning("You need to log in first.")
    else:
        st.subheader("Upload Noisy Image for Denoising and Visualization")
        uploaded = st.file_uploader("Upload Noisy Image", type=["jpg", "png", "jpeg"])
        if uploaded:
            file_id = str(uuid.uuid4())
            noisy_path = f"uploads/{file_id}_{uploaded.name}"
            with open(noisy_path, "wb") as f:
                f.write(uploaded.read())

            st.image(noisy_path, caption="Original Noisy Image", use_column_width=True)

            if st.button("Denoise Image and Visualize Patches"):
                clean_img = simple_denoise(noisy_path)
                clean_path = f"clean/{file_id}_clean.png"
                cv2.imwrite(clean_path, clean_img)

                original = cv2.imread(noisy_path)
                original = cv2.resize(original, (clean_img.shape[1], clean_img.shape[0]))
                psnr_score = psnr(original, clean_img)
                ssim_score = ssim(original, clean_img, channel_axis=-1)

                st.subheader("📊 Denoising Metrics")
                st.metric("PSNR", f"{psnr_score:.2f}")
                st.metric("SSIM", f"{ssim_score:.4f}")

                upload_id = save_upload(st.session_state.user_id, noisy_path)
                save_clean_image(upload_id, clean_path, psnr_score, ssim_score)
                st.success("Denoised image and results saved.")

                st.subheader("🔍 Noisy vs Denoised Image Patches")

                col1, col2 = st.columns(2)
                with col1:
                    st.text("Noisy Patches")
                with col2:
                    st.text("Denoised Patches")

                # Patch parameters
                img_patches = 6
                patch_size = 64
                for i in range(img_patches):
                    x = np.random.randint(0, original.shape[1] - patch_size)
                    y = np.random.randint(0, original.shape[0] - patch_size)

                    noisy_patch = original[y:y+patch_size, x:x+patch_size]
                    clean_patch = clean_img[y:y+patch_size, x:x+patch_size]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(noisy_patch, caption=f"Noisy Patch {i+1}", use_column_width=True)
                    with col2:
                        st.image(clean_patch, caption=f"Denoised Patch {i+1}", use_column_width=True)

                st.subheader("🖼️ Full Image Comparison")
                st.image([noisy_path, clean_path], caption=["Noisy Image", "Denoised Image"], width=300)




# Upload & Denoise
elif choice == "Upload & Denoise":
    if not st.session_state.logged_in:
        st.warning("You need to log in first.")
    else:
        st.subheader("Upload Noisy Image for Denoising")
        uploaded = st.file_uploader("Upload Noisy Image", type=["jpg", "png", "jpeg"])
        if uploaded:
            # Save uploaded file
            file_id = str(uuid.uuid4())
            noisy_path = f"uploads/{file_id}_{uploaded.name}"
            with open(noisy_path, "wb") as f:
                f.write(uploaded.read())

            st.image(noisy_path, caption="Noisy Image")

            if st.button("Denoise Image"):
                # Denoise the image
                clean_img = simple_denoise(noisy_path)
                clean_path = f"clean/{file_id}_clean.png"
                cv2.imwrite(clean_path, clean_img)

                # Calculate metrics
                original = cv2.imread(noisy_path)
                original = cv2.resize(original, (clean_img.shape[1], clean_img.shape[0]))
                psnr_score = psnr(original, clean_img)
                ssim_score = ssim(original, clean_img, channel_axis=-1)

                st.image(clean_path, caption="Denoised Image", use_column_width=True)
                st.metric("PSNR", f"{psnr_score:.2f}")
                st.metric("SSIM", f"{ssim_score:.4f}")

                upload_id = save_upload(st.session_state.user_id, noisy_path)
                save_clean_image(upload_id, clean_path, psnr_score, ssim_score)
                st.success("Denoised image and results saved to the database.")
